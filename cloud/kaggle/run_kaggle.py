"""Thin Kaggle entry point. The repository remains the source of truth."""

import os
from pathlib import Path
import subprocess
import sys


REPO_URL = os.environ.get("THINKINGCAP_REPO", "https://github.com/marek-krcal/ThinkingCap-Test.git")
REPO_REF = os.environ.get("THINKINGCAP_REF", "main")
MODE = os.environ.get("THINKINGCAP_MODE", "smoke")  # smoke or full
# Empty selects the regular 30-question smoke test; use 10 for a scored
# development slice of ten questions per benchmark (40 total).
LIMIT_PER_BENCHMARK = os.environ.get("THINKINGCAP_LIMIT_PER_BENCHMARK", "")
CHECKOUT = Path("/kaggle/temp/ThinkingCap-Test")


def run(command, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(item) for item in command], cwd=cwd, check=True)


if CHECKOUT.exists():
    raise SystemExit(f"Refusing to overwrite existing checkout: {CHECKOUT}")
if MODE not in ("smoke", "full"):
    raise SystemExit("THINKINGCAP_MODE must be smoke or full")
if LIMIT_PER_BENCHMARK and (not LIMIT_PER_BENCHMARK.isdigit() or int(LIMIT_PER_BENCHMARK) < 1):
    raise SystemExit("THINKINGCAP_LIMIT_PER_BENCHMARK must be a positive integer")
if MODE == "full" and LIMIT_PER_BENCHMARK:
    raise SystemExit("Full mode and a development question limit are mutually exclusive")
run(["git", "clone", "--depth", "1", "--no-checkout", REPO_URL, CHECKOUT])
run(["git", "fetch", "--depth", "1", "origin", REPO_REF], CHECKOUT)
run(["git", "checkout", "--detach", "FETCH_HEAD"], CHECKOUT)
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-local.txt"], CHECKOUT)
run([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0), 'PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"], CHECKOUT)

output = "/kaggle/working/runs/base" if MODE == "full" else "/kaggle/working/runs/smoke"

cloud_runner = CHECKOUT / "cloud/run_eval.py"
if cloud_runner.exists():
    command = [sys.executable, "cloud/run_eval.py", "--out", output,
               "--dtype", "half", "--backend", "transformers", "--device", "cuda"]
    if LIMIT_PER_BENCHMARK:
        command += ["--limit-per-benchmark", LIMIT_PER_BENCHMARK]
    if MODE == "full":
        command.append("--full")
    run(command, CHECKOUT)
else:
    raise SystemExit("Selected commit must include cloud/run_eval.py; use this fork's setup branch")
