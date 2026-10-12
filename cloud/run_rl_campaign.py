"""Two <=20-minute training probes, then the fixed 400x1 evaluation each."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run(command, timeout, environment=None):
    print('+', ' '.join(command), flush=True)
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    try:
        status = process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if status:
        raise subprocess.CalledProcessError(status, command)


def main():
    def terminate(*unused):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, terminate)
    root = Path('/kaggle/working/runs/binary-rl-overfit')
    root.mkdir(parents=True, exist_ok=True)
    summary = {'contaminated_test_overfit': True, 'algorithms': {}}
    for algorithm in ('simpo', 'rloo'):
        started = time.monotonic()
        training = root/algorithm/'training'
        evaluation = root/algorithm/'evaluation'
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
        try:
            # 19-minute trainer alarm includes loading/rollouts/updates; final
            # adapter save has 60 seconds' headroom before the outer hard cap.
            run([sys.executable, 'train_rl.py', '--algorithm', algorithm, '--out', str(training),
                 '--allow-test-overfit', '--seconds', '1140', '--max-steps', '100',
                 '--questions-per-benchmark', '4', '--max-new-tokens', '1024',
                 '--group-size', '4'], 1200, environment)
            run([sys.executable, 'cloud/run_eval.py', '--out', str(evaluation),
                 '--adapter', str(training/'adapter'), '--limit-per-benchmark', '100',
                 '--samples', '1'], 1200)
            result = json.loads((evaluation/'results.json').read_text())
            if result['generation']['responses'] != 400 or result['failed_benchmarks']:
                raise RuntimeError('Evaluation incomplete or graders failed')
            summary['algorithms'][algorithm] = {'status': 'completed',
                     'elapsed_seconds': time.monotonic()-started}
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as error:
            summary['algorithms'][algorithm] = {'status': 'failed', 'error': str(error),
                     'elapsed_seconds': time.monotonic()-started}
            print('Algorithm failed:', algorithm, error, flush=True)
        finally:
            (root/'campaign-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    if not all(value['status']=='completed' for value in summary['algorithms'].values()):
        raise RuntimeError('At least one training/evaluation probe failed; inspect campaign summary')


if __name__ == '__main__':
    main()
