"""One <=20-minute RLOO repeat, then matched 400x1 evaluation if trained cleanly."""
import json
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud.run_rl_campaign import run


def main():
    def terminate(*unused):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, terminate)
    root = Path('/kaggle/working/runs/rloo-batched')
    root.mkdir(parents=True, exist_ok=True)
    training, evaluation = root/'training', root/'evaluation'
    summary = dict(contaminated_test_overfit=True, algorithm='rloo', status='starting')
    started = time.monotonic()
    try:
        run([sys.executable, 'train_batched_rl.py', '--algorithm', 'rloo', '--out', str(training),
             '--scratch', '/kaggle/temp/thinkingcap/rl-snapshots', '--allow-test-overfit',
             '--seconds', '1140'], 1200)
        trained = json.loads((training/'training-summary.json').read_text())
        if trained['steps'] < 1 or trained['skipped_nonfinite_updates'] or trained['status'] == 'failed':
            raise RuntimeError('No clean trained checkpoint')
        import torch
        from safetensors.torch import load_file
        tensors = load_file(str(training/'adapter/adapter_model.safetensors'))
        if not all(torch.isfinite(t).all().item() for t in tensors.values()) or not any(
                t.abs().sum().item() > 0 for name, t in tensors.items() if 'lora_B' in name):
            raise RuntimeError('Checkpoint is nonfinite or untrained')
        summary['training_steps'] = trained['steps']
        summary['adapter_sha256'] = trained['adapter_sha256']
        run([sys.executable, 'cloud/run_eval.py', '--out', str(evaluation),
             '--adapter', str(training/'adapter'), '--limit-per-benchmark', '100', '--samples', '1'], 1200)
        result = json.loads((evaluation/'results.json').read_text())
        records = [json.loads(line) for line in (evaluation/'records.jsonl').read_text().splitlines()]
        if len(records) != 400 or len({(r['id'], r['sample_index']) for r in records}) != 400:
            raise RuntimeError('Missing/duplicate evaluation answers')
        if result['failed_benchmarks'] or any(type(r.get('correct')) is not bool or r.get('grading_error') for r in records):
            raise RuntimeError('Evaluation grader failure')
        if any(r.get('adapter_route_verified_by') != 'engine_request_state' or r.get('adapter_id') != 1 or
               r.get('completion_verified_by') != 'vllm_finish_reason' or
               (r['truncated'] and r['correct']) or (r.get('boundary_error') and r['correct']) for r in records):
            raise RuntimeError('Invalid route, completion metadata, or malformed-answer grade')
        summary['status'] = 'completed'
        summary['evaluation_answers'] = len(records)
    except BaseException as error:
        summary['status'] = 'failed'
        summary['error'] = repr(error)
        raise
    finally:
        summary['elapsed_seconds'] = time.monotonic()-started
        (root/'campaign-summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
