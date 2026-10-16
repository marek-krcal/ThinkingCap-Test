"""Build and submit one private collection shard with a durable safety ledger.

This is an explicit dispatcher, not a polling loop. The in-chat monitor chooses
the next unfinished shard, inspects its logs, and calls this only when safe.
"""
import argparse
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


def build_bundle(folder, job_id, attempt, commit, plan_hash, previous, generation_seconds, timeout):
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Pin a full commit ID')
    if not re.fullmatch(r'(test|train|validation)-\d{3}', job_id) or attempt < 1:
        raise ValueError('Invalid job/attempt')
    if not 60 <= generation_seconds <= timeout-600 or timeout > 10800:
        raise ValueError('Reserve at least ten minutes for setup, grading and archival')
    folder.mkdir(parents=True, exist_ok=False)
    title = f'ThinkingCap offline {job_id} a{attempt}'
    slug = title.lower().replace(' ', '-')
    kernel = 'krcalkrcal/'+slug
    environment = dict(THINKINGCAP_REF=commit, THINKINGCAP_MODE='full', THINKINGCAP_SAMPLES='8',
        THINKINGCAP_OFFLINE_JOB=job_id, THINKINGCAP_GENERATION_SECONDS=str(generation_seconds),
        THINKINGCAP_TIMEOUT_SECONDS=str(timeout), THINKINGCAP_PLAN_SHA=plan_hash)
    # Build artifacts: this tiny launcher is generated from fixed fields. There
    # are no credentials, user-code interpolation, or shell substitutions.
    source = ('import os, urllib.request\n'
        f'os.environ.update({environment!r})\n'
        f'url="https://raw.githubusercontent.com/marek-krcal/ThinkingCap-Test/{commit}/cloud/kaggle-vllm/run.py"\n'
        'with urllib.request.urlopen(url, timeout=60) as response:\n'
        '    source=response.read().decode("utf-8")\n'
        'exec(compile(source,url,"exec"), {"__name__":"__main__", "__file__":url})\n')
    (folder/'run.py').write_text(source)
    metadata = dict(id=kernel, title=title, code_file='run.py', language='python',
        kernel_type='script', is_private='true', enable_gpu='true', enable_internet='true',
        dataset_sources=['krcalkrcal/thinkingcap-offline-input'], competition_sources=[],
        kernel_sources=['krcalkrcal/thinkingcap-vllm-timing']+([previous] if previous else []),
        model_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return kernel


def check_capacity(api, ledger, timeout):
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
                    '/thinkingcap-offline-cpu-verification' in row.ref):
                continue
            status = retry_read(lambda: api.kernels_status(row.ref)).status.name
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
    options = parser.parse_args()
    data = Path(options.data)
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
    state = ROOT/'runs/offline-campaign'
    state.mkdir(parents=True, exist_ok=True)
    with (state/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = state/'ledger.json'
        ledger = json.loads(path.read_text()) if path.exists() else dict(plan_sha256=plan_hash, attempts=[])
        if ledger['plan_sha256'] != plan_hash:
            raise ValueError('Ledger belongs to another frozen plan')
        previous = [a for a in ledger['attempts'] if a['job_id']==options.job]
        parent = None
        if previous:
            latest = previous[-1]
            if latest['state'] != 'partial':
                raise ValueError('Previous attempt must be audited partial before resuming; do not duplicate work')
            parent = latest['kernel']
        attempt = len(previous)+1
        folder = state/'submissions'/f'{options.job}-a{attempt}'
        if not options.submit:
            print(json.dumps(dict(job=options.job, attempt=attempt, parent=parent, commit=options.commit,
                plan_sha256=plan_hash, timeout=options.timeout, generation_seconds=options.generation_seconds),indent=2))
            return
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        quota = check_capacity(api, ledger, options.timeout)
        if parent and api.kernels_status(parent).status.name != 'COMPLETE':
            raise RuntimeError('Previous output must be COMPLETE and attachable, not running or failed')
        kernel = build_bundle(folder, options.job, attempt, options.commit, plan_hash,
            parent, options.generation_seconds, options.timeout)
        row = dict(job_id=options.job, attempt=attempt, kernel=kernel, commit=options.commit,
            timeout=options.timeout, submitted_at=time.time(), state='submission_pending', quota_before=quota,
            completed_answers=previous[-1]['completed_answers'] if previous else 0,
            new_answers=0, parent_kernel=parent)
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
