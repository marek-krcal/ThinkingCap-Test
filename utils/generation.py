"""Shared durable continuous queue for evaluation and reusable offline data."""
import json
import time


def run_queue(engine, jobs, settings_for, record_for, path, lora=None, validate_route=None,
              deadline=None):
    wanted = {f'eval-{i}': job for i, job in enumerate(jobs)}
    for request_id, job in wanted.items():
        engine.add_request(request_id, {'prompt_token_ids': job['ids']},
                           settings_for(job), lora_request=lora)
        if validate_route:
            validate_route(request_id)
    completed = {}
    with path.open('w') as stream:
        while engine.has_unfinished_requests():
            if deadline is not None and time.time() >= deadline:
                for request_id in wanted.keys()-completed.keys():
                    engine.abort_request(request_id)
                break
            for output in engine.step():
                if not output.finished:
                    continue
                request_id = output.request_id
                if request_id not in wanted or request_id in completed:
                    raise RuntimeError('Unknown or duplicate continuous-queue answer')
                record = record_for(wanted[request_id], output, request_id)
                completed[request_id] = record
                stream.write(json.dumps(record)+'\n')
                stream.flush()
                if len(completed) % 20 == 0 or len(completed) == len(jobs):
                    print(f'Continuous queue: {len(completed)}/{len(jobs)} completed', flush=True)
    if deadline is None and set(completed) != set(wanted):
        raise RuntimeError('Continuous queue lost generation jobs')
    return [completed[request_id] for request_id in wanted if request_id in completed]
