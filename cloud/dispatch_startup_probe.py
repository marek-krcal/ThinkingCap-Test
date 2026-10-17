"""Submit one durable CPU-only startup diagnostic, never automatically retry."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloud.dispatch_offline import check_capacity, verify_resume_cache
from utils.offline import digest


def build_probe(folder):
    folder.mkdir(parents=True, exist_ok=False)
    source = (ROOT/'cloud/offline_startup_probe.py').read_text()
    compile(source, 'run.py', 'exec')
    (folder/'run.py').write_text(source)
    metadata = dict(id='krcalkrcal/thinkingcap-offline-startup-diagnostic',
        title='ThinkingCap offline startup diagnostic', code_file='run.py',
        language='python', kernel_type='script', is_private='true',
        enable_gpu='false', enable_internet='true',
        dataset_sources=['krcalkrcal/thinkingcap-offline-subset-input'],
        kernel_sources=['krcalkrcal/thinkingcap-offline-subset-train-004-a2'],
        competition_sources=[], model_sources=[])
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata['id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--verified-inactive-draft', action='append', default=[])
    args = parser.parse_args()
    if not args.submit:
        print('CPU-only input/fetch diagnostic; pass --submit once after fresh checks')
        return
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    state = ROOT/'runs/offline-startup-diagnostic'
    state.mkdir(parents=True, exist_ok=True)
    with (ROOT/'runs/offline-campaign/dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger_file = state/'ledger.json'
        if ledger_file.exists():
            raise RuntimeError('Diagnostic intent already exists; resolve exact session, never blindly retry')
        attempts = []
        for path in (ROOT/'runs/offline-campaign/ledger.json', ROOT/'runs/offline-subset-campaign/ledger.json'):
            attempts.extend(json.loads(path.read_text())['attempts'])
        quota = check_capacity(api, {'attempts': attempts}, 600, args.verified_inactive_draft)
        if quota['active']:
            raise RuntimeError('Diagnostic requires idle GPU collectors')
        page = 1
        while True:
            rows = api.kernels_list(mine=True, page=page, page_size=100) or []
            for row in rows:
                if '/thinkingcap-offline-' in row.ref:
                    if api.kernels_status(row.ref).status.name in ('RUNNING', 'QUEUED', 'CANCEL_REQUESTED'):
                        raise RuntimeError('Wait for active campaign CPU/session: '+row.ref)
            if len(rows) < 100:
                break
            page += 1
        parent = 'krcalkrcal/thinkingcap-offline-subset-train-004-a2'
        if api.kernels_status(parent).status.name != 'COMPLETE':
            raise RuntimeError('Diagnostic parent is not COMPLETE')
        verify_resume_cache(api, parent)
        folder = state/'submission'
        kernel = build_probe(folder)
        row = dict(kernel=kernel, state='submission_pending', submitted_at=time.time(),
                   timeout=600, gpu=False, parent_kernel=parent, quota_before=quota,
                   source_sha256=digest(folder/'run.py'))
        ledger_file.write_text(json.dumps(row, indent=2)+'\n')
        response = api.kernels_push(str(folder), timeout='600')
        row.update(state='submitted', url=response.url, version=response.version_number)
        ledger_file.write_text(json.dumps(row, indent=2)+'\n')
        print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
