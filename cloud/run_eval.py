"""Cloud evaluation with a fixed, measured two-T4 inference recipe."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runs/cloud-evaluation')
    parser.add_argument('--model', default='Qwen/Qwen3-0.6B')
    parser.add_argument('--adapter')
    parser.add_argument('--samples', type=int, choices=(1, 2, 4, 8), default=1,
                        help='Answers per question')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--limit-per-benchmark', type=int, choices=(100, 200),
                           help='Questions per benchmark; omitted means full coverage')
    selection.add_argument('--full', action='store_true', help='Explicit full coverage')
    selection.add_argument('--smoke', action='store_true', help='30-question plumbing check')
    parser.add_argument('--benchmarks', default='mbpp,bfcl,svamp,ifeval')
    return parser.parse_args()


def evaluation_command(args):
    command = [sys.executable, 'evaluate.py', '--out', args.out,
               '--model', args.model, '--samples', str(args.samples),
               '--benchmarks', args.benchmarks, '--backend', 'vllm',
               '--dtype', 'half', '--tensor-parallel-size', '1',
               '--data-parallel-size', '2', '--continuous-queue',
               '--max-num-seqs', '64', '--max-num-batched-tokens', '2048',
               '--gpu-memory-utilization', '0.70', '--cuda-graphs']
    if args.adapter:
        command += ['--adapter', args.adapter]
    if args.limit_per_benchmark:
        command += ['--limit-per-benchmark', str(args.limit_per_benchmark)]
    elif args.smoke:
        command.append('--smoke')
    return command


def main():
    args = parse_arguments()
    if not (ROOT / 'questions/manifest.json').exists():
        subprocess.run([sys.executable, 'utils/eval/benchmarks.py'], cwd=ROOT, check=True)
    command = evaluation_command(args)
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
