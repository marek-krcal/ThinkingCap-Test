"""Thin Kaggle entry point. The repository remains the source of truth."""

import os
from pathlib import Path
import subprocess
import sys


REPO_URL = os.environ.get("THINKINGCAP_REPO", "https://github.com/marek-krcal/ThinkingCap-Test.git")
REPO_REF = os.environ.get("THINKINGCAP_REF", "main")
MODE = os.environ.get("THINKINGCAP_MODE", "smoke")  # smoke or full
# The first cloud run is intentionally bounded: ten questions from each of
# MBPP, BFCL, SVAMP and IFEval (40 total). Set this empty for the regular smoke
# mode or override it for another development slice.
LIMIT_PER_BENCHMARK = os.environ.get("THINKINGCAP_LIMIT_PER_BENCHMARK", "10")
CHECKOUT = Path("/kaggle/working/ThinkingCap-Test")


def run(command, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(item) for item in command], cwd=cwd, check=True)


if CHECKOUT.exists():
    raise SystemExit(f"Refusing to overwrite existing checkout: {CHECKOUT}")
run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, CHECKOUT])
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], CHECKOUT)

output = "/kaggle/working/runs/base" if MODE == "full" else "/kaggle/working/runs/smoke"
if MODE not in ("smoke", "full"):
    raise SystemExit("THINKINGCAP_MODE must be smoke or full")
if LIMIT_PER_BENCHMARK and (not LIMIT_PER_BENCHMARK.isdigit() or int(LIMIT_PER_BENCHMARK) < 1):
    raise SystemExit("THINKINGCAP_LIMIT_PER_BENCHMARK must be a positive integer")

cloud_runner = CHECKOUT / "cloud/run_eval.py"
if cloud_runner.exists():
    command = [sys.executable, "cloud/run_eval.py", "--out", output, "--dtype", "auto"]
    if LIMIT_PER_BENCHMARK:
        command += ["--limit-per-benchmark", LIMIT_PER_BENCHMARK]
    if MODE == "full":
        command.append("--full")
    run(command, CHECKOUT)
else:
    # This makes the unmodified upstream repository usable on free T4/P100 GPUs.
    # A fork containing this setup takes the preferred branch above.
    evaluation = CHECKOUT / "evaluate.py"
    source = evaluation.read_text()
    needle = "tokenizer_revision=arguments.revision,\n"
    if needle not in source:
        raise SystemExit("Could not apply the upstream fp16 compatibility patch")
    evaluation.write_text(source.replace(needle, needle + "                dtype=\"half\",\n", 1))
    run([sys.executable, "utils/eval/benchmarks.py"], CHECKOUT)
    command = [sys.executable, "evaluate.py", "--out", output]
    if MODE == "smoke":
        command += ["--smoke", "--samples", "1"]
    run(command, CHECKOUT)
