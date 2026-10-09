"""Isolated CUDA 12.8 vLLM comparison, with explicit setup timings."""
import json
import pathlib
import subprocess
import sys
import time

root = pathlib.Path('/kaggle/working/ThinkingCap-Test')
timings = {}

def run(label, command, cwd=None):
    started = time.perf_counter()
    print(label, command, flush=True)
    try:
        subprocess.run(command, cwd=cwd, check=True)
    finally:
        timings[label] = time.perf_counter() - started
        pathlib.Path('/kaggle/working/setup-timing.json').write_text(json.dumps(timings, indent=2))
        print(label, 'seconds:', timings[label], flush=True)

run('clone', ['git', 'clone', '--depth', '1', 'https://github.com/marek-krcal/ThinkingCap-Test.git', str(root)])
run('installer_bootstrap', [sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
run('python_environment', ['uv', 'venv', '--python', '3.12', '/kaggle/working/vllm-env'])
python = '/kaggle/working/vllm-env/bin/python'
run('dependency_install', ['uv', 'pip', 'install', '--python', python, '--torch-backend', 'cu128', 'vllm==0.11.2', '-r', 'requirements-local.txt'], root)
run('import_check', [python, '-c', "import torch, vllm; print('torch',torch.__version__,'CUDA',torch.version.cuda,'vllm',vllm.__version__); assert torch.cuda.is_available()"], root)
run('evaluation_total', [python, 'cloud/run_eval.py', '--out', '/kaggle/working/runs/vllm', '--backend', 'vllm', '--dtype', 'half', '--limit-per-benchmark', '10', '--samples', '1'], root)
