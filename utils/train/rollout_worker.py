"""Persistent single-GPU vLLM worker, refreshed LoRA snapshot each round.

JSON RPC on stdin; result messages are prefixed so vLLM stdout logs cannot be
mistaken for responses. No optimizer, rewards, or stale-policy replay here.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

PREFIX = 'RL_RPC:'


def sampling_kwargs(seed, cap, eos_ids):
    return dict(temperature=1., top_p=1., top_k=-1, min_p=0.,
                repetition_penalty=1., presence_penalty=0., frequency_penalty=0.,
                seed=seed, max_tokens=cap, ignore_eos=True,
                stop_token_ids=list(eos_ids), logprobs=1)


def unpack_completion(completion, eos_ids, cap):
    tokens = list(completion.token_ids)
    if completion.finish_reason not in ('stop', 'length'):
        raise ValueError('Unexpected rollout completion reason')
    if not tokens or len(tokens) > cap:
        raise ValueError('Empty/over-cap rollout')
    if completion.finish_reason == 'stop' and tokens[-1] not in eos_ids:
        raise ValueError('Stopped rollout lacks the scored EOS action')
    if completion.logprobs is None or len(completion.logprobs) != len(tokens):
        raise ValueError('Missing selected-token sampling probabilities')
    logps = [float(row[token].logprob) for token, row in zip(tokens, completion.logprobs)]
    return dict(completion_tokens=tokens, sampler_logprobs=logps,
                finish_reason=completion.finish_reason, stop_reason=completion.stop_reason,
                generation_cap=cap, ceiling_hit=len(tokens) == cap)


def run_round(model, request, previous_adapter):
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import RequestOutputKind
    started = time.monotonic()
    engine = model.llm_engine
    if engine.has_unfinished_requests():
        raise RuntimeError('Cannot change policy while requests remain')
    if previous_adapter is not None and not engine.remove_lora(previous_adapter):
        raise RuntimeError('Old LoRA could not be removed')
    weights = Path(request['adapter'])/'adapter_model.safetensors'
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != request['adapter_sha256']:
        raise RuntimeError('Snapshot hash mismatch')
    adapter_id = request['round']+1
    lora = LoRARequest(f'policy-{adapter_id}', adapter_id, request['adapter'])
    if not engine.add_lora(lora):
        raise RuntimeError('New LoRA could not be registered')
    sync_seconds = time.monotonic()-started
    jobs = request['jobs']
    pending, answers, cursor = {}, {}, 0
    started = time.monotonic()
    while cursor < len(jobs) or pending:
        while cursor < len(jobs) and len(pending) < 64:
            job = jobs[cursor]
            rid = f"round-{request['round']}-sample-{cursor}"
            params = SamplingParams(**sampling_kwargs(job['seed'], request['cap'], request['eos_ids']),
                                    output_kind=RequestOutputKind.FINAL_ONLY)
            engine.add_request(rid, {'prompt_token_ids': job['prompt_tokens']}, params, lora_request=lora)
            state = engine.output_processor.request_states.get(rid)
            if state is None or state.lora_name != lora.lora_name:
                raise RuntimeError('Adapter route mismatch at engine input')
            pending[rid] = cursor
            cursor += 1
        for output in engine.step():
            if not output.finished:
                continue
            index = pending.pop(output.request_id)
            answers[index] = {**unpack_completion(output.outputs[0], set(request['eos_ids']), request['cap']),
                              'adapter_id': adapter_id,
                              'adapter_route_verified_by': 'engine_request_state',
                              'adapter_sha256': digest}
            if len(answers) % 32 == 0:
                print(f'Rollout round {request["round"]}: {len(answers)}/{len(jobs)} completed', flush=True)
    return dict(answers=[answers[i] for i in range(len(jobs))], sync_seconds=sync_seconds,
                generation_seconds=time.monotonic()-started, adapter_id=adapter_id), adapter_id


def main():
    from vllm import LLM
    from evaluate import BASE_MODEL, BASE_REVISION
    started = time.monotonic()
    model = LLM(model=BASE_MODEL, revision=BASE_REVISION, dtype='half',
                generation_config='vllm', enforce_eager=True, tensor_parallel_size=1,
                enable_lora=True, max_lora_rank=16, max_loras=1, max_cpu_loras=1,
                max_model_len=5120, max_num_seqs=64, max_num_batched_tokens=2048,
                gpu_memory_utilization=.70, enable_prefix_caching=False)
    print(PREFIX+json.dumps({'ready': True, 'startup_seconds': time.monotonic()-started}), flush=True)
    adapter = None
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('close'):
            break
        try:
            result, adapter = run_round(model, request, adapter)
            print(PREFIX+json.dumps(result), flush=True)
        except Exception as error:
            print(PREFIX+json.dumps({'error': repr(error)}), flush=True)
            raise


if __name__ == '__main__':
    main()
