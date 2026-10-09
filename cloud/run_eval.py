"""Prepare the frozen benchmark cohort and run a reproducible cloud evaluation."""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(item) for item in command], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/cloud-smoke")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter")
    parser.add_argument("--backend", choices=("auto", "vllm", "transformers"), default="auto")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "half", "bfloat16", "float"), default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--full", action="store_true", help="Run the full eight-sample evaluation")
    parser.add_argument("--samples", type=int, help="Override samples (smoke defaults to 1)")
    parser.add_argument("--limit-per-benchmark", type=int,
                        help="Development only: evaluate this many questions per benchmark")
    parser.add_argument("--benchmarks", default="mbpp,bfcl,svamp,ifeval")
    args = parser.parse_args()

    if not (ROOT / "questions" / "manifest.json").exists():
        run([sys.executable, "utils/eval/benchmarks.py"])

    command = [sys.executable, "evaluate.py", "--out", args.out,
               "--model", args.model, "--dtype", args.dtype,
               "--benchmarks", args.benchmarks,
               "--backend", args.backend, "--device", args.device]
    command += ["--tensor-parallel-size", str(args.tensor_parallel_size),
                "--batch-size", str(args.batch_size)]
    if args.adapter:
        command += ["--adapter", args.adapter]
    if args.limit_per_benchmark:
        command += ["--limit-per-benchmark", str(args.limit_per_benchmark)]
    if args.limit_per_benchmark:
        # A benchmark-sized development slice must come from the frozen full
        # cohort; the 30-question plumbing smoke set is deliberately smaller.
        command += ["--samples", str(args.samples or 1)]
    elif not args.full:
        command += ["--smoke", "--samples", str(args.samples or 1)]
    elif args.samples:
        command += ["--samples", str(args.samples)]
    run(command)


if __name__ == "__main__":
    main()
