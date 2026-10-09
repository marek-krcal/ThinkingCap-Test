"""Isolated CUDA 12.8 vLLM comparison, with explicit setup timings."""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

root = pathlib.Path('/kaggle/working/ThinkingCap-Test')
timings = {}
cache = pathlib.Path('/kaggle/working/eval-cache')
cache.mkdir(exist_ok=True)
os.environ['NLTK_DATA'] = str(cache / 'nltk_data')
(cache / 'nltk_data').mkdir(mode=0o700, exist_ok=True)

def run(label, command, cwd=None):
    started = time.perf_counter()
    print(label, command, flush=True)
    try:
        subprocess.run(command, cwd=cwd, check=True)
    finally:
        timings[label] = time.perf_counter() - started
        pathlib.Path('/kaggle/working/setup-timing.json').write_text(json.dumps(timings, indent=2))
        print(label, 'seconds:', timings[label], flush=True)

def configure_cuda_linker():
    """Expose the real host driver library under its compiler link name."""
    listing = subprocess.check_output(['ldconfig', '-p'], text=True)
    candidates = [pathlib.Path(line.split('=>', 1)[1].strip())
                  for line in listing.splitlines()
                  if 'libcuda.so.1 ' in line and '=>' in line]
    for directory in ('/usr/lib/x86_64-linux-gnu', '/usr/local/nvidia/lib64',
                      '/usr/lib64-nvidia', '/usr/lib64'):
        candidates.extend(pathlib.Path(directory).glob('libcuda.so.1'))
    driver = next((p.resolve() for p in candidates if p.is_file() and 'stubs' not in str(p)), None)
    if driver is None:
        raise RuntimeError('Cannot locate host libcuda.so.1 for FlashInfer linking')
    link_dir = pathlib.Path('/kaggle/working/cuda-driver-link')
    link_dir.mkdir(exist_ok=True)
    (link_dir / 'libcuda.so').symlink_to(driver)
    (link_dir / 'libcuda.so.1').symlink_to(driver)
    for variable in ('LIBRARY_PATH', 'LD_LIBRARY_PATH'):
        os.environ[variable] = str(link_dir) + ':' + os.environ.get(variable, '')
    print('CUDA driver linker path:', driver, flush=True)
    # Fail cheaply before package installation if the compiler still cannot link.
    subprocess.run(['cc', '-x', 'c', '-', '-lcuda', '-o', str(link_dir / 'link-check')],
                   input='extern int cuInit(unsigned); int main(void) { return cuInit(0); }',
                   text=True, check=True)
    subprocess.run([str(link_dir / 'link-check')], check=True)

configure_cuda_linker()

run('clone', ['git', 'clone', '--depth', '1', 'https://github.com/marek-krcal/ThinkingCap-Test.git', str(root)])
run('installer_bootstrap', [sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
run('python_environment', ['uv', 'venv', '--python', '3.12', '/kaggle/working/vllm-env'])
python = '/kaggle/working/vllm-env/bin/python'
run('dependency_install', ['uv', 'pip', 'install', '--python', python, '--torch-backend', 'cu128', 'vllm==0.11.2', 'nltk==3.9.2', '-r', 'requirements-local.txt'], root)
run('import_check', [python, '-c', "import torch, vllm; print('torch',torch.__version__,'CUDA',torch.version.cuda,'vllm',vllm.__version__); assert torch.cuda.is_available()"], root)
# Attach an earlier successful kernel's output to reuse its frozen cohort.
for manifest in pathlib.Path('/kaggle/input').rglob('eval-cache/questions/manifest.json'):
    shutil.copytree(manifest.parent, root / 'questions')
    previous_nltk = manifest.parent.parent / 'nltk_data'
    if previous_nltk.exists():
        shutil.copytree(previous_nltk, cache / 'nltk_data', dirs_exist_ok=True)
    print('Restored evaluation cache:', manifest.parent, flush=True)
    break
if not (root / 'questions/manifest.json').exists():
    run('benchmark_preparation', [python, 'utils/eval/benchmarks.py'], root)
shutil.copytree(root / 'questions', cache / 'questions', dirs_exist_ok=True)
run('evaluation_total', [python, 'cloud/run_eval.py', '--out', '/kaggle/working/runs/vllm', '--backend', 'vllm', '--dtype', 'half', '--limit-per-benchmark', '10', '--samples', '1'], root)
