"""Short binary test-overfitting probes with fresh batched vLLM rollouts.

RLOO: 32 prompt instances x8, exactly one accumulated update/round.
SimPO: 64 prompt instances x8, up to eight disjoint pair minibatches/round.
The original tiny train_rl.py remains an unchanged performance reference.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace

import torch
from evaluate import (BASE_MODEL, BASE_REVISION, BENCHMARKS, PROMPT_BUDGET,
                      load_suite, read_jsonl, limit_questions_per_benchmark,
                      grading_context, grade_all, record_from_tokens)
from train_rl import BudgetExpired
from utils.train.chunked import score_completion
from utils.train.objectives import binary_rewards, leave_one_out, simpo_loss
from utils.train.rollout_client import RolloutClient


def build_round(questions, tokenizer, round_id, prompts, group_size, seed):
    jobs = []
    for group in range(prompts):
        question = questions[(round_id*prompts+group) % len(questions)]
        prompt = tokenizer.apply_chat_template(question['messages'], tokenize=True,
                     add_generation_prompt=True, enable_thinking=True)
        if not 0 < len(prompt) <= PROMPT_BUDGET:
            raise ValueError('Training prompt exceeds budget')
        for sample in range(group_size):
            jobs.append(dict(row=question, group_index=group, sample=sample,
                seed=seed+round_id*100000+group*group_size+sample, prompt_tokens=prompt))
    return jobs


def group_advantages(records, group_size):
    if len(records) % group_size:
        raise ValueError('Incomplete prompt group')
    advantages, kinds = [], Counter()
    for start in range(0, len(records), group_size):
        group = records[start:start+group_size]
        if len({r['group_index'] for r in group}) != 1:
            raise ValueError('Mixed prompt groups')
        rewards = binary_rewards(group)
        kinds['all_correct' if all(rewards) else 'all_incorrect' if not any(rewards) else 'informative'] += 1
        advantages.extend(leave_one_out(torch.tensor(rewards)).tolist())
    return advantages, dict(kinds)


def simpo_pairs(records, group_size):
    pairs = []
    for start in range(0, len(records), group_size):
        group = records[start:start+group_size]
        winners = [start+i for i, r in enumerate(group) if r['reward'] == 1 and not r['truncated']]
        losers = [start+i for i, r in enumerate(group) if r['reward'] == 0 and not r['truncated']]
        # Binary-only probe: no brevity pair, no correlated all-pairs expansion.
        if winners and losers:
            pairs.append((winners[0], losers[0]))
    return pairs


def pair_batches(pairs, count=8):
    return [pairs[i::min(count, len(pairs))] for i in range(min(count, len(pairs)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=('simpo', 'rloo'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--allow-test-overfit', action='store_true', required=True)
    parser.add_argument('--seconds', type=int, default=1140)
    args = parser.parse_args()
    if not 0 < args.seconds <= 1140:
        parser.error('Each training probe must fit its 19-minute alarm')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Training output must be fresh')
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started+args.seconds
    model = worker = None
    prompts, size, cap = (32 if args.algorithm == 'rloo' else 64), 8, 1024
    summary = dict(algorithm=args.algorithm, contaminated_test_overfit=True,
        reward='binary_correctness_only', length_penalty=0, kl_penalty=0,
        model=BASE_MODEL, revision=BASE_REVISION, sampling=dict(temperature=1., top_p=1., top_k=0),
        dtype='float16', lora_rank=16, learning_rate=1e-4, seed=42, generation_cap=cap,
        prompt_instances_per_round=prompts, group_size=size, trajectories_per_round=prompts*size,
        optimizer_updates_per_round=1 if args.algorithm == 'rloo' else 8,
        loss_backend='checkpointed_chunked_lm_head_128', steps=0, rounds=0,
        generated_trajectories=0, generated_tokens=0, ceiling_hits=0, truncations=0,
        skipped_nonfinite_updates=0, budget_seconds=args.seconds, status='starting')

    def emit(event, **data):
        row = dict(event=event, elapsed_seconds=time.monotonic()-started,
                   time=time.time(), policy_step=summary['steps'], **data)
        print(json.dumps(row), flush=True)
        with (args.out/'training-events.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')

    def expire(*unused):
        raise BudgetExpired()

    def persist():
        if model is not None:
            model.save_pretrained(args.out/'adapter')
            weights = args.out/'adapter/adapter_model.safetensors'
            summary['adapter_sha256'] = hashlib.sha256(weights.read_bytes()).hexdigest()
        summary['elapsed_seconds'] = time.monotonic()-started
        (args.out/'training-summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    def finish_update(optimizer, scaler, loss_value, batch_count):
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = scaler.get_scale() < scale
        summary['skipped_nonfinite_updates'] += int(skipped)
        summary['steps'] += int(not skipped)
        optimizer.zero_grad(set_to_none=True)
        emit('update', loss=loss_value, gradient_norm=float(norm), skipped_nonfinite=skipped,
             effective_batch=batch_count)

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    random.seed(42)
    torch.manual_seed(42)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        suite = load_suite(Path('questions'), False)
        selected = limit_questions_per_benchmark(read_jsonl(Path('questions/questions.jsonl')),
                                                list(BENCHMARKS), 4)
        pools = [[q for q in selected if q['dataset'] in ds] for ds in BENCHMARKS.values()]
        questions = [pool[i] for i in range(4) for pool in pools]
        summary['question_ids'] = [q['id'] for q in questions]
        summary['cohort_sha256'] = suite['cohort_sha256']
        (args.out/'training-question-ids.json').write_text(json.dumps(summary['question_ids']))
        grading = grading_context(suite, SimpleNamespace(sandbox=None, workers=4, benchmarks=list(BENCHMARKS)))
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
        eos_ids = {tokenizer.eos_token_id}  # same stop policy as the original training probe
        open_id, close_id = [tokenizer.convert_tokens_to_ids(tag) for tag in ('<think>', '</think>')]
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION,
                   dtype=torch.float16, attn_implementation='sdpa').to('cuda:0')
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
                   target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
        scaler = torch.amp.GradScaler('cuda', init_scale=128, growth_interval=100)
        emit('model_loaded', trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        worker = RolloutClient(1, deadline, args.out/'rollout-worker.log')
        emit('rollout_engine_ready', **worker.receive())
        while summary['steps'] < 100 and deadline-time.monotonic() > 120:
            round_id = summary['rounds']
            emit('round_started', round=round_id)
            tick = time.monotonic()
            snapshot = args.scratch/f'snapshot-{round_id}'
            model.save_pretrained(snapshot)
            digest = hashlib.sha256((snapshot/'adapter_model.safetensors').read_bytes()).hexdigest()
            jobs = build_round(questions, tokenizer, round_id, prompts, size, 42)
            save_seconds = time.monotonic()-tick
            if round_id == 0:
                # Fail cheaply on CUDA/route/sampling incompatibility before
                # spending a full 256/512-trajectory generation round.
                probe = worker.generate(dict(round=round_id, adapter=str(snapshot.resolve()), adapter_sha256=digest,
                    cap=16, eos_ids=list(eos_ids), jobs=[{k: j[k] for k in ('prompt_tokens', 'seed')} for j in jobs[:2]]))
                model.eval()
                differences = []
                with torch.no_grad():
                    for job, answer in zip(jobs[:2], probe['answers']):
                        actual = score_completion(model, job['prompt_tokens'], answer['completion_tokens'], True)
                        sampled = torch.tensor(answer['sampler_logprobs'], device=actual.device)
                        differences.extend((actual-sampled).abs().cpu().tolist())
                mean_diff, max_diff = sum(differences)/len(differences), max(differences)
                emit('gpu_preflight', probe_trajectories=2, probe_cap=16,
                     mean_absolute_logprob_difference=mean_diff, max_absolute_logprob_difference=max_diff)
                if mean_diff > .05 or max_diff > .5:
                    raise RuntimeError('Sampler/learner preflight probability mismatch')
            reply = worker.generate(dict(round=round_id, adapter=str(snapshot.resolve()), adapter_sha256=digest,
                cap=cap, eos_ids=list(eos_ids), jobs=[{k: j[k] for k in ('prompt_tokens', 'seed')} for j in jobs]))
            emit('generation_finished', round=round_id, generation_seconds=reply['generation_seconds'],
                 sync_seconds=reply['sync_seconds']+save_seconds, responses=len(reply['answers']))
            if len(reply['answers']) != len(jobs):
                raise ValueError('Incomplete generation round')
            records = []
            for job, answer in zip(jobs, reply['answers']):
                tokens = answer['completion_tokens']
                record = record_from_tokens(job, tokens, tokenizer, eos_ids, open_id, close_id)
                record.update(answer, prompt_tokens=job['prompt_tokens'], round=round_id,
                    group_index=job['group_index'], policy_step=summary['steps'],
                    truncated=answer['finish_reason'] != 'stop')
                if record['adapter_sha256'] != digest or record['adapter_id'] != round_id+1:
                    raise RuntimeError('Rollout snapshot mismatch')
                records.append(record)
            # Diagnostic only: no gradient, no update before this consistency check.
            model.eval()
            differences = []
            with torch.no_grad():
                for record in records[:2]:
                    actual = score_completion(model, record['prompt_tokens'], record['completion_tokens'], True)
                    sampled = torch.tensor(record['sampler_logprobs'], device=actual.device)
                    differences.extend((actual-sampled).abs().cpu().tolist())
            mean_diff, max_diff = sum(differences)/len(differences), max(differences)
            emit('sampling_probability_check', round=round_id, tokens=len(differences),
                 mean_absolute_logprob_difference=mean_diff, max_absolute_logprob_difference=max_diff)
            if mean_diff > .05 or max_diff > .5:
                raise RuntimeError('Sampler/learner probability mismatch beyond FP16 tolerance')
            tick = time.monotonic()
            grade_all(questions, records, grading)
            rewards = binary_rewards(records)
            for record, reward in zip(records, rewards):
                record['reward'] = reward
            advantages, kinds = group_advantages(records, size)
            generated_tokens = sum(r['output_tokens'] for r in records)
            summary['rounds'] += 1
            summary['generated_trajectories'] += len(records)
            summary['generated_tokens'] += generated_tokens
            summary['ceiling_hits'] += sum(r['ceiling_hit'] for r in records)
            summary['truncations'] += sum(r['truncated'] for r in records)
            with (args.out/'rollouts.jsonl').open('a') as stream:
                for record in records:
                    stream.write(json.dumps(record)+'\n')
            emit('rollout_graded', round=round_id, grading_seconds=time.monotonic()-tick,
                 groups=kinds, reward_mean=sum(rewards)/len(rewards), generated_tokens=generated_tokens,
                 ceiling_hits=sum(r['ceiling_hit'] for r in records), trajectories=len(records),
                 ceiling_hit_fraction=sum(r['ceiling_hit'] for r in records)/len(records),
                 cumulative_ceiling_hit_fraction=summary['ceiling_hits']/summary['generated_trajectories'],
                 truncated=sum(r['truncated'] for r in records),
                 tokens_per_generation_second=generated_tokens/reply['generation_seconds'])
            tick = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if args.algorithm == 'rloo':
                loss_value = 0.
                used = 0
                for record, advantage in zip(records, advantages):
                    # Zero-advantage groups are exact zero terms, not a new denominator.
                    if advantage == 0:
                        continue
                    loss = -advantage*score_completion(model, record['prompt_tokens'], record['completion_tokens'])/len(records)
                    loss_value += float(loss.detach())
                    scaler.scale(loss).backward()
                    used += 1
                    if used % 16 == 0:
                        emit('gradient_accumulation', round=round_id, trajectories_processed=used)
                if used:
                    finish_update(optimizer, scaler, loss_value, len(records))
                else:
                    emit('zero_advantage_round_skipped', round=round_id)
            else:
                pairs = simpo_pairs(records, size)
                random.shuffle(pairs)
                for batch in pair_batches(pairs):
                    optimizer.zero_grad(set_to_none=True)
                    loss_value = 0.
                    for i, j in batch:
                        with torch.no_grad():
                            sums = [score_completion(model, records[k]['prompt_tokens'], records[k]['completion_tokens']) for k in (i, j)]
                        leaves = [x.detach().float().requires_grad_() for x in sums]
                        loss = simpo_loss(*leaves, len(records[i]['completion_tokens']), len(records[j]['completion_tokens']))/len(batch)
                        loss.backward()
                        loss_value += float(loss.detach())
                        for leaf, k in zip(leaves, (i, j)):
                            scaler.scale(score_completion(model, records[k]['prompt_tokens'], records[k]['completion_tokens'])*leaf.grad.detach()).backward()
                    finish_update(optimizer, scaler, loss_value, len(batch))
                emit('pair_round_finished', round=round_id, pairs=len(pairs), updates=len(pair_batches(pairs)))
            emit('round_finished', round=round_id, update_seconds=time.monotonic()-tick)
            persist()
        summary['status'] = 'budget_reached' if summary['steps'] else 'no_learning_signal'
    except (BudgetExpired, TimeoutError):
        summary['status'] = 'budget_reached' if summary['steps'] else 'no_learning_signal'
        emit('budget_reached', incomplete_round_not_updated=True)
    except Exception as error:
        summary['status'] = 'failed'
        summary['error'] = repr(error)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if worker is not None:
            worker.close()
        persist()
    if not summary['steps'] or summary['skipped_nonfinite_updates']:
        raise RuntimeError('No clean completed optimizer update; do not evaluate an untrained checkpoint')


if __name__ == '__main__':
    main()
