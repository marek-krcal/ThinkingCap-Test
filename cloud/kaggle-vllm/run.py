"""Isolated CUDA 12.8 vLLM comparison, with explicit setup timings."""
import json
import datetime
import os
import pathlib
import signal
import shutil
import subprocess
import sys
import threading
import time

# Kaggle archives everything in /kaggle/working. Keep disposable multi-GB
# environments, driver links and source checkout out of the output tree.
scratch = pathlib.Path('/kaggle/temp/thinkingcap')
scratch.mkdir(parents=True, exist_ok=True)
root = scratch / 'ThinkingCap-Test'
os.environ['UV_CACHE_DIR'] = str(scratch / 'uv-cache')
os.environ['HF_HOME'] = str(scratch / 'huggingface')
# Kaggle's paired T4s have no usable GPU P2P link. Force the host transport.
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'
timings = {}
cutoff = datetime.datetime(2026, 10, 10, 0, tzinfo=datetime.timezone.utc).timestamp()
hard_stop = cutoff + 7200
cache = pathlib.Path('/kaggle/working/eval-cache')
cache.mkdir(exist_ok=True)
os.environ['NLTK_DATA'] = str(cache / 'nltk_data')
(cache / 'nltk_data').mkdir(mode=0o700, exist_ok=True)

def run(label, command, cwd=None, timeout=1500):
    started = time.perf_counter()
    print(label, command, flush=True)
    try:
        process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
        try:
            returncode = process.wait(timeout=min(timeout, max(1, hard_stop - time.time())))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
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
    link_dir = scratch / 'cuda-driver-link'
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
run('python_environment', ['uv', 'venv', '--python', '3.12', str(scratch / 'vllm-env')])
python = str(scratch / 'vllm-env/bin/python')
run('dependency_install', ['uv', 'pip', 'install', '--python', python, '--torch-backend', 'cu128', 'vllm==0.11.2', 'nltk==3.9.2', '-r', 'requirements-local.txt'], root)
run('import_check', [python, '-c', "import torch, vllm; print('torch',torch.__version__,'CUDA',torch.version.cuda,'vllm',vllm.__version__); print('GPUs:',torch.cuda.device_count()); assert torch.cuda.device_count() >= 2, 'Two GPUs required'"], root)
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

stopped = threading.Event()
def monitor_gpus():
    """Log each GPU's real utilization so allocation is not mistaken for use."""
    while not stopped.is_set():
        try:
            stats = subprocess.check_output([
                'nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'], text=True, timeout=10)
            print('GPU index, utilization %, memory used MiB, total MiB:\n' + stats,
                  flush=True)
            with pathlib.Path('/kaggle/working/gpu-utilization.jsonl').open('a') as stream:
                stream.write(json.dumps({'time': time.time(), 'stats': stats}) + '\n')
        except (OSError, subprocess.SubprocessError) as error:
            print('GPU telemetry unavailable:', error, flush=True)
        stopped.wait(15)

monitor = threading.Thread(target=monitor_gpus, daemon=True)
monitor.start()
try:
    # User revised the campaign to full x1 after v6 completed the 400-question
    # stage. Do not spend quota repeating that completed stage.
    if time.time() < cutoff:
        run('full_x1_evaluation_total', [python, 'cloud/run_eval.py',
            '--out', '/kaggle/working/runs/full-x1', '--backend', 'vllm', '--dtype', 'half',
            '--full', '--samples', '1', '--tensor-parallel-size', '2',
            '--batch-size', '512'], root, timeout=max(1, hard_stop - time.time()))
    else:
        print('Cutoff reached: full evaluation will not be started.', flush=True)
finally:
    stopped.set()
    monitor.join(timeout=12)
print('Finished; output contains only results, timings, telemetry and benchmark cache.', flush=True)
