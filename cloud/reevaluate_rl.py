"""Evaluate saved binary-overfit adapters; never train or alter their weights."""
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

from cloud.run_rl_campaign import run

EXPECTED = {
    'simpo': '8a322c33ada3fc86266cad663258d3f3372dc9348ca805d25f24eb1696b373b4',
    'rloo': '30acb51cc197a3d0816580dbe2c8ce0304fa6e9f7276824b9a5cf4decd229105',
}


def find_adapter(inputs, algorithm):
    candidates = list(inputs.rglob(f'runs/binary-rl-overfit/{algorithm}/training/adapter/adapter_model.safetensors'))
    matches = [p for p in candidates if hashlib.sha256(p.read_bytes()).hexdigest() == EXPECTED[algorithm]]
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one hash-verified saved {algorithm} adapter; got {len(matches)}')
    adapter = matches[0].parent
    training = json.loads((adapter.parent/'training-summary.json').read_text())
    if training['steps'] < 1 or training['skipped_nonfinite_updates']:
        raise RuntimeError('Saved checkpoint did not complete clean training')
    return adapter, training


def main():
    def terminate(*unused):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, terminate)
    root = Path('/kaggle/working/runs/binary-rl-evaluation')
    root.mkdir(parents=True, exist_ok=True)
    summary = {'evaluation_only': True, 'contaminated_test_overfit': True, 'algorithms': {}}
    for algorithm in ('simpo', 'rloo'):
        adapter, training = find_adapter(Path('/kaggle/input'), algorithm)
        started = time.monotonic()
        out = root/algorithm
        print('Hash-verified saved adapter:', algorithm, adapter, flush=True)
        # Fail fast on any remaining routing incompatibility; avoid launching
        # a second doomed evaluation. Completed first results stay preserved.
        run([sys.executable, 'cloud/run_eval.py', '--out', str(out), '--adapter', str(adapter),
             '--limit-per-benchmark', '100', '--samples', '1'], 1200)
        result = json.loads((out/'results.json').read_text())
        records = [json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
        if len(records) != 400 or len({(r['id'], r['sample_index']) for r in records}) != 400:
            raise RuntimeError('Incomplete or duplicate evaluation answers')
        if result['failed_benchmarks'] or any(r.get('grading_error') or type(r.get('correct')) is not bool for r in records):
            raise RuntimeError('Grading failed')
        if any(r.get('adapter_route_verified_by') != 'engine_request_state' or r.get('adapter_id') != 1 for r in records):
            raise RuntimeError('Missing verified adapter route')
        summary['algorithms'][algorithm] = {'status': 'completed', 'adapter_sha256': EXPECTED[algorithm],
              'training_steps': training['steps'], 'evaluation_seconds': time.monotonic()-started,
              'answers': len(records)}
        (root/'evaluation-summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
