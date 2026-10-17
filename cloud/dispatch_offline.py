"""Build and submit one private collection shard with a durable safety ledger.

This is an explicit dispatcher, not a polling loop. The in-chat monitor chooses
the next unfinished shard, inspects its logs, and calls this only when safe.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.offline import digest


def retry_read(operation):
    """Retry only harmless reads; never blindly retry a GPU submission."""
    import requests
    for attempt in range(3):
        try:
            return operation()
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 2: raise
            time.sleep(attempt+1)


def build_bundle(folder, job_id, attempt, commit, plan_hash, previous, generation_seconds, timeout,
                 dataset='krcalkrcal/thinkingcap-offline-input', prefix='ThinkingCap offline',
                 minimal_resume_inputs=False, notebook_launcher=False):
    if minimal_resume_inputs and not previous:
        raise ValueError('Minimal resume inputs require an audited parent')
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Pin a full commit ID')
    if not re.fullmatch(r'(test|train|validation)-\d{3}', job_id) or attempt < 1:
        raise ValueError('Invalid job/attempt')
    if not 60 <= generation_seconds <= timeout-600 or timeout > 10800:
        raise ValueError('Reserve at least ten minutes for setup, grading and archival')
    folder.mkdir(parents=True, exist_ok=False)
    if not re.fullmatch(r'ThinkingCap [a-z -]+', prefix):
        raise ValueError('Invalid collection namespace')
    title = f'{prefix} {job_id} a{attempt}'
    slug = title.lower().replace(' ', '-')
    kernel = 'krcalkrcal/'+slug
    environment = dict(THINKINGCAP_REF=commit, THINKINGCAP_MODE='full', THINKINGCAP_SAMPLES='8',
        THINKINGCAP_OFFLINE_JOB=job_id, THINKINGCAP_GENERATION_SECONDS=str(generation_seconds),
        THINKINGCAP_OFFLINE_DATASET=dataset,
        THINKINGCAP_TIMEOUT_SECONDS=str(timeout), THINKINGCAP_PLAN_SHA=plan_hash)
    # Build artifacts: this tiny launcher is generated from fixed fields. There
    # are no credentials, user-code interpolation, or shell substitutions.
    source = ('import os, subprocess, sys\n'
        f'print("Collection launcher starting: {job_id} attempt {attempt}", flush=True)\n'
        f'os.environ.update({environment!r})\n'
        f'url="https://raw.githubusercontent.com/marek-krcal/ThinkingCap-Test/{commit}/cloud/kaggle-vllm/run.py"\n'
        'fetch="import sys,urllib.request; r=urllib.request.urlopen(sys.argv[1],timeout=60); sys.stdout.buffer.write(r.read())"\n'
        'source=subprocess.run([sys.executable,"-c",fetch,url],check=True,stdout=subprocess.PIPE,timeout=90).stdout.decode("utf-8")\n'
        'print("Pinned launcher fetched; starting setup", flush=True)\n'
        'exec(compile(source,url,"exec"), {"__name__":"__main__", "__file__":url})\n')
    (folder/'run.py').write_text(source)
    if notebook_launcher:
        notebook = dict(nbformat=4, nbformat_minor=5,
            metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
            cells=[dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                        source=source.splitlines(keepends=True))])
        (folder/'run.ipynb').write_text(json.dumps(notebook, indent=2)+'\n')
    metadata = dict(id=kernel, title=title,
        code_file='run.ipynb' if notebook_launcher else 'run.py', language='python',
        kernel_type='notebook' if notebook_launcher else 'script',
        is_private='true', enable_gpu='true', enable_internet='true',
        dataset_sources=[dataset], competition_sources=[],
        kernel_sources=([previous] if minimal_resume_inputs else
            ['krcalkrcal/thinkingcap-vllm-timing']+([previous] if previous else [])),
        model_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return kernel


def verify_resume_cache(api, parent):
    """Inspect filenames only; never download raw outputs or environments."""
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    owner, slug = parent.split('/')
    required = {'eval-cache/questions/manifest.json', 'collection/collection-index.json'}
    found, token, seen = set(), None, set()
    while True:
        request = ApiListKernelSessionOutputRequest()
        request.user_name, request.kernel_slug, request.page_size = owner, slug, 200
        if token:
            request.page_token = token
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
        found.update(item.file_name for item in response.files or [])
        token = response.next_page_token
        if not token:
            break
        if token in seen:
            raise RuntimeError('Repeated parent output page token')
        seen.add(token)
    if not required.issubset(found):
        raise RuntimeError('Minimal resume parent must contain its cache manifest and collection index')


def resume_parent(previous):
    if not previous: return None
    latest=previous[-1]
    if latest['state']=='partial': return latest
    if (latest['state']=='cancelled_startup' and latest.get('new_answers')==0 and
            latest.get('remote_status')=='CANCEL_ACKNOWLEDGED' and latest.get('verified_output_bytes')==0):
        audited=[a for a in previous[:-1] if a['state'] in ('partial','completed')]
        if audited:
            parent=audited[-1]
            if (parent['state']=='partial' and latest.get('parent_kernel')==parent['kernel'] and
                    latest.get('completed_answers')==parent['completed_answers']):
                return parent
    raise ValueError('Previous attempt must be audited partial or verified stopped zero-output startup; do not duplicate work')


def check_capacity(api, ledger, timeout, verified_inactive_drafts=()):
    with api.build_kaggle_client() as client:
        quota = retry_read(client.kernels.kernels_api_client.get_accelerator_quota_statistics)
    gpu = quota.gpu_quota
    if gpu is None or gpu.total_time_allowed is None or gpu.time_used is None or gpu.time_reserved is None:
        raise RuntimeError('Actual Kaggle quota unavailable; do not guess capacity')
    active = []
    listed = set()
    page = 1
    while True:
        rows = retry_read(lambda: api.kernels_list(mine=True, page=page, page_size=100)) or []
        for row in rows:
            listed.add(row.ref)
            # Kaggle's lightweight listing can omit enable_gpu and deserialize
            # it as False even for a T4 run. Exclude only our known CPU launchers.
            if ('/thinkingcap-offline-grade-' in row.ref or
                    '/thinkingcap-offline-cpu-verification' in row.ref or
                    '/thinkingcap-offline-subset-reconcile' in row.ref):
                continue
            try:
                status = retry_read(lambda: api.kernels_status(row.ref)).status.name
            except Exception as error:
                # Draft-only notebooks have no version session endpoint. Never
                # infer inactivity from a 404: require a fresh editor check and
                # an explicit per-invocation acknowledgement for this exact ref.
                if (getattr(getattr(error, 'response', None), 'status_code', None) == 404
                        and row.ref in verified_inactive_drafts
                        and getattr(row, 'current_version_number', None) in (None, 0)):
                    continue
                raise
            if status in ('RUNNING', 'QUEUED', 'CANCEL_REQUESTED'):
                active.append(row.ref)
        if len(rows) < 100: break
        page += 1
    if any(a.get('state')=='submission_pending' and a['kernel'] not in listed for a in ledger['attempts']):
        raise RuntimeError('Resolve an ambiguous submission before launching more jobs')
    for attempt in ledger['attempts']:
        if attempt.get('state') == 'submitted' and attempt['kernel'] not in listed:
            # A new push may not appear in the list immediately. Treat it as
            # occupying a slot until a refresh proves its terminal outcome.
            active.append(attempt['kernel'])
    if len(active) >= 2:
        raise RuntimeError('Two account sessions already active/queued: '+str(active))
    # Account quota can lag queued work. Reserve its entire declared timeout
    # locally as well, without double-subtracting server reservations.
    outstanding = sum(a['timeout'] for a in ledger['attempts'] if a['kernel'] in active)
    seconds = gpu.total_time_allowed.total_seconds()-gpu.time_used.total_seconds()
    seconds -= max(gpu.time_reserved.total_seconds(), outstanding)
    if seconds < timeout+120:
        raise RuntimeError(f'Wait for quota reset: only {seconds:.0f}s unreserved; need {timeout+120}s')
    return dict(used=gpu.time_used.total_seconds(), reserved=gpu.time_reserved.total_seconds(),
        allowed=gpu.total_time_allowed.total_seconds(), active=active,
        verified_inactive_drafts=list(verified_inactive_drafts),
        refresh=str(quota.quota_refresh_time))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--data', default='data/offline-input')
    parser.add_argument('--verification', default='runs/kaggle-offline-verification-v1/runs/offline-verification/verification-summary.json')
    parser.add_argument('--generation-seconds', type=int, default=2700)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--state', default='runs/offline-campaign')
    parser.add_argument('--dataset', default='krcalkrcal/thinkingcap-offline-input')
    parser.add_argument('--prefix', default='ThinkingCap offline')
    parser.add_argument('--minimal-resume-inputs', action='store_true',
        help='Omit redundant timing notebook only after verifying the audited parent contains cache/index')
    parser.add_argument('--notebook-launcher', action='store_true',
        help='Deliver the identical pinned launcher in one notebook cell instead of script mode')
    parser.add_argument('--verified-inactive-draft', action='append', default=[],
        help='Exact draft-only ref just verified OFF in its editor; not a persistent exemption')
    options = parser.parse_args()
    data = Path(options.data)
    selection_file = ROOT/'data/offline-subset/selection.json'
    if options.job.startswith('train-') and selection_file.exists():
        selection = json.loads(selection_file.read_text())
        if digest(data/'train.jsonl') != selection['train_sha256']:
            raise RuntimeError('Training scope changed to the selected subset: migrate its plan/ledger and reuse saved answers before dispatch')
        if (options.state=='runs/offline-campaign' or options.prefix=='ThinkingCap offline' or
                options.dataset=='krcalkrcal/thinkingcap-offline-input'):
            raise RuntimeError('Selected training requires its own state, notebook namespace and private input')
    plan_file = data/'collection-plan.json'
    plan_hash = digest(plan_file)
    plan = json.loads(plan_file.read_text())
    if options.job not in {j['id'] for j in plan['jobs']}:
        raise ValueError('Unknown shard')
    verification = json.loads(Path(options.verification).read_text())
    if (verification.get('status') != 'passed' or verification.get('answers') != 400 or
            not verification.get('repeated_grading_identical') or
            verification.get('regenerated_for_grading') != 0):
        raise ValueError('400-answer offline verification has not passed')
    state = ROOT/options.state
    state.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        global_state=ROOT/'runs/offline-campaign';global_state.mkdir(parents=True,exist_ok=True)
        for lock_path in sorted({global_state/'dispatch.lock',state/'dispatch.lock'}):
            lock=stack.enter_context(lock_path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = state/'ledger.json'
        ledger = json.loads(path.read_text()) if path.exists() else dict(plan_sha256=plan_hash, attempts=[])
        if (options.job.startswith('train-') and selection_file.exists() and
                ledger.get('reconcile_state')!='audited'):
            raise RuntimeError('Selected training requires the audited CPU reuse bootstrap before generation')
        if ledger.get('reuse_pending'):
            raise RuntimeError('Reconcile completed old outputs before generating selected training')
        if ledger['plan_sha256'] != plan_hash:
            raise ValueError('Ledger belongs to another frozen plan')
        previous = [a for a in ledger['attempts'] if a['job_id']==options.job]
        parent = None
        if previous:
            parent = resume_parent(previous)['kernel']
        attempt = len(previous)+1
        folder = state/'submissions'/f'{options.job}-a{attempt}'
        if not options.submit:
            print(json.dumps(dict(job=options.job, attempt=attempt, parent=parent, commit=options.commit,
                plan_sha256=plan_hash, timeout=options.timeout, generation_seconds=options.generation_seconds),indent=2))
            return
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        if previous and previous[-1]['state']=='cancelled_startup':
            if api.kernels_status(previous[-1]['kernel']).status.name!='CANCEL_ACKNOWLEDGED':
                raise RuntimeError('Cancelled startup must still be verified stopped before replacement')
        capacity_attempts=list(ledger['attempts'])
        for peer in (ROOT/'runs/offline-campaign/ledger.json',ROOT/'runs/offline-subset-campaign/ledger.json'):
            if peer.exists() and peer.resolve()!=path.resolve():
                capacity_attempts.extend(json.loads(peer.read_text())['attempts'])
        quota = check_capacity(api, dict(attempts=capacity_attempts), options.timeout, options.verified_inactive_draft)
        if parent and api.kernels_status(parent).status.name != 'COMPLETE':
            raise RuntimeError('Previous output must be COMPLETE and attachable, not running or failed')
        if options.minimal_resume_inputs:
            if not parent:
                raise RuntimeError('Minimal resume inputs require an audited parent')
            verify_resume_cache(api, parent)
        kernel = build_bundle(folder, options.job, attempt, options.commit, plan_hash,
            parent, options.generation_seconds, options.timeout, options.dataset, options.prefix,
            options.minimal_resume_inputs, options.notebook_launcher)
        row = dict(job_id=options.job, attempt=attempt, kernel=kernel, commit=options.commit,
            timeout=options.timeout, submitted_at=time.time(), state='submission_pending', quota_before=quota,
            completed_answers=previous[-1]['completed_answers'] if previous else 0,
            new_answers=0, parent_kernel=parent,
            minimal_resume_inputs=options.minimal_resume_inputs,
            notebook_launcher=options.notebook_launcher)
        ledger['attempts'].append(row)
        # Record intent first. An ambiguous network failure must not cause a
        # second submission; the monitor resolves the known kernel identity.
        path.write_text(json.dumps(ledger, indent=2)+'\n')
        response = api.kernels_push(str(folder), timeout=str(options.timeout), acc='NvidiaTeslaT4')
        row.update(state='submitted', url=response.url, version=response.version_number)
        path.write_text(json.dumps(ledger, indent=2)+'\n')
        print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
