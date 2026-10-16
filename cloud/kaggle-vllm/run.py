"""Isolated CUDA 12.8 vLLM comparison, with explicit setup timings."""
import json
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
# Fresh, bounded run on any date. Defaults to a 30-question plumbing smoke test.
# Edit these defaults before pushing, or set THINKINGCAP_* in the runtime.
repo = os.environ.get('THINKINGCAP_REPO', 'https://github.com/marek-krcal/ThinkingCap-Test.git')
ref = os.environ.get('THINKINGCAP_REF', 'feature/kaggle-evaluation')  # exact commit recommended
mode = os.environ.get('THINKINGCAP_MODE', 'smoke')
samples = int(os.environ.get('THINKINGCAP_SAMPLES', '1'))
limit = int(os.environ.get('THINKINGCAP_LIMIT_PER_BENCHMARK', '0'))
tp, dp = 1, 2  # measured two-T4 recipe; not runtime configuration
timeout = int(os.environ.get('THINKINGCAP_TIMEOUT_SECONDS', '7200'))
if mode not in ('smoke', 'full') or samples not in (1, 2, 4, 8) or timeout < 1 or limit not in (0, 100, 200):
    raise ValueError('Invalid run configuration')
if mode == 'full' and limit:
    raise ValueError('Full coverage and a development question limit are mutually exclusive')
hard_stop = time.time() + timeout
cache = pathlib.Path('/kaggle/working/eval-cache')
cache.mkdir(exist_ok=True)
os.environ['NLTK_DATA'] = str(cache / 'nltk_data')
(cache / 'nltk_data').mkdir(mode=0o700, exist_ok=True)

def run(label, command, cwd=None, timeout=1500):
    remaining = hard_stop - time.time()
    if remaining <= 0:
        raise TimeoutError('Total setup/evaluation time limit reached')
    started = time.perf_counter()
    print(label, command, flush=True)
    try:
        process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
        try:
            returncode = process.wait(timeout=min(timeout, remaining))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if returncode:
            # A failed parent can leave vLLM worker processes alive.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
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

run('clone', ['git', 'clone', '--depth', '1', '--no-checkout', repo, str(root)])
run('fetch_ref', ['git', 'fetch', '--depth', '1', 'origin', ref], root)
run('checkout_ref', ['git', 'checkout', '--detach', 'FETCH_HEAD'], root)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
pathlib.Path('/kaggle/working/run-config.json').write_text(json.dumps({
    'repo': repo, 'ref': ref, 'commit': commit, 'mode': mode, 'samples': samples,
    'limit_per_benchmark': limit, 'tensor_parallel_size': tp, 'data_parallel_size': dp,
    'offline_job': os.environ.get('THINKINGCAP_OFFLINE_JOB'),
    'collection_plan_sha256': os.environ.get('THINKINGCAP_PLAN_SHA'),
    'dtype': 'half', 'max_num_seqs': 64, 'continuous_queue': True,
    'timeout_seconds': timeout}, indent=2))
run('installer_bootstrap', [sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
run('python_environment', ['uv', 'venv', '--python', '3.12', str(scratch / 'vllm-env')])
python = str(scratch / 'vllm-env/bin/python')
run('dependency_install', ['uv', 'pip', 'install', '--python', python, '--torch-backend', 'cu128', 'vllm==0.11.2', 'nltk==3.9.2', '-r', 'requirements-local.txt'], root)
run('import_check', [python, '-c', "import torch, vllm; print('torch',torch.__version__,'CUDA',torch.version.cuda,'vllm',vllm.__version__); print('GPUs:',torch.cuda.device_count()); assert torch.cuda.device_count() >= " + str(tp * dp)], root)
with pathlib.Path('/kaggle/working/installed-packages.txt').open('w') as stream:
    subprocess.run(['uv', 'pip', 'freeze', '--python', python], stdout=stream, check=True)
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
        stopped.wait(1 if os.environ.get('THINKINGCAP_BATCHED_RL') == '1' else 15)

monitor = threading.Thread(target=monitor_gpus, daemon=True)
monitor.start()
try:
    command = [python, 'cloud/run_eval.py', '--out', '/kaggle/working/runs/evaluation',
        '--samples', str(samples)]
    if limit:
        command += ['--limit-per-benchmark', str(limit)]
    elif mode == 'full':
        command.append('--full')
    else:
        command.append('--smoke')
    if os.environ.get('THINKINGCAP_BINARY_RL') == '1':
        command = [python, 'cloud/run_rl_campaign.py']
    if os.environ.get('THINKINGCAP_REEVAL_BINARY_RL') == '1':
        command = [python, 'cloud/reevaluate_rl.py']
    if os.environ.get('THINKINGCAP_BATCHED_RL') == '1':
        command = [python, 'cloud/run_batched_rl_campaign.py']
    if os.environ.get('THINKINGCAP_OFFLINE_VERIFY') == '1':
        command = [python, 'cloud/verify_offline.py']
    if os.environ.get('THINKINGCAP_OFFLINE_JOB'):
        data_manifest = next(pathlib.Path('/kaggle/input').rglob('collection-plan.json'))
        command = [python, 'cloud/collect_offline.py', '--data', str(data_manifest.parent),
            '--job', os.environ['THINKINGCAP_OFFLINE_JOB'],
            '--expected-plan-sha', os.environ['THINKINGCAP_PLAN_SHA'],
            '--generation-seconds', os.environ.get('THINKINGCAP_GENERATION_SECONDS', '2700')]
        resume_indexes = list(pathlib.Path('/kaggle/input').rglob('collection-index.json'))
        if len(resume_indexes) > 1:
            raise ValueError('Attach at most one previous attempt for this shard')
        if resume_indexes:
            command += ['--resume-index', str(resume_indexes[0])]
    run('evaluation_total', command, root, timeout=max(1, hard_stop - time.time()))
finally:
    stopped.set()
    monitor.join(timeout=12)
print('Finished; output contains only results, timings, telemetry and benchmark cache.', flush=True)
