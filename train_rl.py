"""Short, explicitly contaminated test-set overfitting probes: SimPO or RLOO.

SimPO collects a fixed preference pool before updating. RLOO samples fresh
groups at every update (one update per group, no stale replay/importance ratio).
The original train.py demonstration remains available unchanged.
"""
import argparse
import json
import random
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from evaluate import (BASE_MODEL, BASE_REVISION, BENCHMARKS, PROMPT_BUDGET,
                      load_suite, read_jsonl, limit_questions_per_benchmark,
                      grading_context, grade_all, record_from_tokens)
from utils.train.objectives import (binary_rewards, preference_pairs,
                                   leave_one_out, simpo_loss, completion_logprob_sum)


class BudgetExpired(Exception):
    pass


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=('simpo', 'rloo'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--questions', type=Path, default=Path('questions'))
    parser.add_argument('--allow-test-overfit', action='store_true', required=True)
    parser.add_argument('--seconds', type=int, default=1140)
    parser.add_argument('--questions-per-benchmark', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=1024)
    parser.add_argument('--group-size', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.seconds <= 1200 or args.group_size < 2 or args.max_new_tokens < 1:
        parser.error('Budget must be 1–1200 seconds and group size >=2')
    if args.questions_per_benchmark < 1 or args.max_steps < 1 or args.learning_rate <= 0:
        parser.error('Question count, steps and learning rate must be positive')
    return args


def main():
    args = parse_arguments()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Training output must be fresh')
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.seconds
    def expire(*unused):
        raise BudgetExpired()
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = None
    summary = {'algorithm': args.algorithm, 'contaminated_test_overfit': True,
               'reward': 'binary_correctness_only', 'length_penalty': 0, 'kl_penalty': 0,
               'steps': 0, 'attempted_updates': 0, 'skipped_nonfinite_updates': 0,
               'groups': 0, 'informative_groups': 0, 'budget_seconds': args.seconds,
               'generation_cap': args.max_new_tokens, 'group_size': args.group_size,
               'sampling': {'temperature': 1.0, 'top_p': 1.0, 'top_k': 0},
               'model': BASE_MODEL, 'revision': BASE_REVISION,
               'dtype': 'float16', 'lora_rank': 16, 'learning_rate': args.learning_rate,
               'seed': args.seed, 'status': 'starting'}
    def emit(event, **details):
        row = {'event': event, 'elapsed_seconds': time.monotonic()-started, **details}
        print(json.dumps(row), flush=True)
        with (args.out/'training-events.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
    def persist():
        if model is not None:
            model.save_pretrained(args.out/'adapter')
        summary['elapsed_seconds'] = time.monotonic()-started
        (args.out/'training-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from peft import LoraConfig, get_peft_model
        suite = load_suite(args.questions, False)
        questions = limit_questions_per_benchmark(read_jsonl(args.questions/'questions.jsonl'),
                                                  list(BENCHMARKS), args.questions_per_benchmark)
        # Interleave benchmarks rather than using up the budget on coding alone.
        pools = [[q for q in questions if q['dataset'] in ds] for ds in BENCHMARKS.values()]
        questions = [pool[i] for i in range(args.questions_per_benchmark)
                     for pool in pools if i < len(pool)]
        (args.out/'training-question-ids.json').write_text(json.dumps([q['id'] for q in questions]))
        summary['question_ids'] = [q['id'] for q in questions]
        summary['cohort_sha256'] = suite['cohort_sha256']
        grading = grading_context(suite, SimpleNamespace(sandbox=None, workers=2,
                                                        benchmarks=list(BENCHMARKS)))
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
        eos = tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        pad = next(iter(eos_ids))
        open_id, close_id = (tokenizer.convert_tokens_to_ids(tag) for tag in ('<think>', '</think>'))
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION,
                      dtype=torch.float16, attn_implementation='sdpa').to('cuda:0')
        model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
                   target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=args.learning_rate, weight_decay=0)
        # Sequence-sum RLOO gradients can be much larger than token-mean SFT
        # gradients; avoid wasting the short run backing off a 65,536 scale.
        scaler = torch.amp.GradScaler('cuda', init_scale=128, growth_interval=100)
        emit('model_loaded', trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))

        def score(prompt, completion, grad=True):
            tokens = torch.tensor([prompt+completion], device='cuda:0')
            # Only completion logits are needed, including first prediction.
            logits = model(tokens, use_cache=False, logits_to_keep=len(completion)+1).logits
            return completion_logprob_sum(logits, tokens[:, len(prompt)-1:], 1)

        def group(question):
            remaining = deadline-time.monotonic()
            if remaining < 60:
                raise BudgetExpired()
            prompt = tokenizer.apply_chat_template(question['messages'], tokenize=True,
                          add_generation_prompt=True, enable_thinking=True)
            if len(prompt) > PROMPT_BUDGET:
                raise ValueError('Training prompt exceeds budget')
            model.eval()
            with torch.no_grad():
                inputs = torch.tensor([prompt]*args.group_size, device='cuda:0')
                # Do not inherit hidden sampling processors from the Hub's
                # generation_config: sampled policy must equal scored policy.
                generation = GenerationConfig(max_new_tokens=args.max_new_tokens,
                          max_time=min(120, remaining-30), do_sample=True,
                          temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                          eos_token_id=eos, pad_token_id=pad, use_cache=True)
                outputs = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                                         generation_config=generation,
                                         temperature=1.0, top_p=1.0, top_k=0,
                                         do_sample=True, repetition_penalty=1.0)
            completions, records = [], []
            for i, output in enumerate(outputs):
                completion = output[len(prompt):].tolist()
                stop = next((j+1 for j, token in enumerate(completion) if token in eos_ids), len(completion))
                completion = completion[:stop]
                completions.append(completion)
                records.append(record_from_tokens({'row': question, 'sample': i,
                                 'seed': args.seed}, completion, tokenizer, eos_ids, open_id, close_id))
            grade_all([question], records, grading)
            rewards = binary_rewards(records)  # fail on infrastructure errors
            summary['groups'] += 1
            informative = bool(preference_pairs(rewards))
            summary['informative_groups'] += int(informative)
            with (args.out/'rollouts.jsonl').open('a') as f:
                for completion, record, reward in zip(completions, records, rewards):
                    f.write(json.dumps({**record, 'reward': reward, 'policy_step': summary['steps'],
                                        'prompt_tokens': prompt, 'completion_tokens': completion})+'\n')
            emit('rollout', question_id=question['id'], rewards=rewards,
                 tokens=[len(c) for c in completions], informative=informative,
                 policy_step=summary['steps'])
            model.train()
            return prompt, completions, rewards

        def update(prompt, completions, rewards, pair=None):
            if deadline-time.monotonic() < 30:
                raise BudgetExpired()
            optimizer.zero_grad(set_to_none=True)
            summary['attempted_updates'] += 1
            if pair is not None:
                # Derive scalar coefficients without retaining two model graphs.
                # Recompute one trajectory at a time: same exact SimPO gradient,
                # much smaller peak memory (dropout is disabled).
                i, j = pair
                model.eval()
                with torch.no_grad():
                    sums = [score(prompt, completions[k]).detach() for k in (i, j)]
                leaves = [x.float().requires_grad_() for x in sums]
                loss = simpo_loss(leaves[0], leaves[1], len(completions[i]), len(completions[j]))
                loss.backward()
                model.train()
                for leaf, k in zip(leaves, (i, j)):
                    scaler.scale(score(prompt, completions[k])*leaf.grad.detach()).backward()
                value = float(loss.detach())
            else:
                advantages = leave_one_out(torch.tensor(rewards, device='cuda:0'))
                value = 0.0
                for completion, advantage in zip(completions, advantages):
                    loss = -advantage * score(prompt, completion) / len(completions)
                    value += float(loss.detach())
                    scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < previous_scale
            summary['skipped_nonfinite_updates'] += int(skipped)
            summary['steps'] += int(not skipped)
            optimizer.zero_grad(set_to_none=True)
            emit('update', step=summary['steps'], loss=value, gradient_norm=float(norm),
                 skipped_nonfinite=skipped, reward_mean=sum(rewards)/len(rewards))
            if summary['steps'] and summary['steps'] % 5 == 0:
                persist()

        if args.algorithm == 'simpo':
            pool = []
            collection_stop = started + min(args.seconds*0.40, 420)
            for question in questions:
                if time.monotonic() > collection_stop:
                    break
                prompt, completions, rewards = group(question)
                pool.extend((prompt, completions, rewards, pair) for pair in preference_pairs(rewards))
            summary['preference_pairs'] = len(pool)
            emit('preference_pool_frozen', pairs=len(pool))
            if not pool:
                summary['status'] = 'no_learning_signal'
            else:
                while summary['attempted_updates'] < args.max_steps and time.monotonic() < deadline-30:
                    update(*random.choice(pool))
                summary['status'] = 'completed'
        else:
            index = 0
            while summary['attempted_updates'] < args.max_steps and time.monotonic() < deadline-60:
                prompt, completions, rewards = group(questions[index % len(questions)])
                index += 1
                if preference_pairs(rewards):
                    update(prompt, completions, rewards)
                else:
                    emit('zero_advantage_group_skipped')
            summary['status'] = 'completed' if summary['steps'] else 'no_learning_signal'
    except BudgetExpired:
        summary['status'] = 'budget_reached' if summary['steps'] else 'no_learning_signal'
        emit('budget_reached')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        persist()
    if model is None or summary['steps'] == 0:
        raise RuntimeError('No optimizer update completed; do not label this a trained checkpoint')


if __name__ == '__main__':
    main()
