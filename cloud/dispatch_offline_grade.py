"""Submit one CPU-only saved-answer grading job; never reserve a GPU."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_launcher(commit):
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Pin a full commit')
    return ('import os, subprocess, sys\n'
        'print("CPU grading launcher starting", flush=True)\n'
        f'os.environ.update(THINKINGCAP_REF={commit!r},THINKINGCAP_TIMEOUT_SECONDS="3600")\n'
        f'url="https://raw.githubusercontent.com/marek-krcal/ThinkingCap-Test/{commit}/cloud/kaggle-cpu-grade.py"\n'
        'fetch="import sys,urllib.request; r=urllib.request.urlopen(sys.argv[1],timeout=60); sys.stdout.buffer.write(r.read())"\n'
        'source=subprocess.run([sys.executable,"-c",fetch,url],check=True,stdout=subprocess.PIPE,timeout=90).stdout.decode("utf-8")\n'
        'print("Pinned CPU launcher fetched; starting setup", flush=True)\n'
        'exec(compile(source,url,"exec"),{"__name__":"__main__","__file__":url})\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--retry', action='store_true', help='Repair a failed CPU grader only')
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--job')
    choice.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}',args.commit): raise ValueError('Pin a full commit')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    state = ROOT/'runs/offline-campaign'; state.mkdir(parents=True,exist_ok=True)
    with (state/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger_file = state/'ledger.json'
        if args.verify:
            title = 'ThinkingCap offline CPU verification'
            source = 'krcalkrcal/thinkingcap-offline-verification'
            ledger, attempt = None, None
        else:
            ledger = json.loads(ledger_file.read_text())
            attempt = [a for a in ledger['attempts'] if a['job_id']==args.job][-1]
            if attempt['state'] not in ('generated','audit_required'):
                raise ValueError('Raw output must be generated, not already graded or submitted')
            if attempt.get('grade_kernel'):
                prior = api.kernels_status(attempt['grade_kernel']).status.name
                if not args.retry or prior not in ('ERROR','CANCEL_ACKNOWLEDGED'):
                    raise ValueError('Retry only an explicitly stopped, failed CPU grader')
                attempt.setdefault('grade_history', []).append(dict(
                    kernel=attempt['grade_kernel'], commit=attempt.get('grade_commit'),
                    remote_status=prior, reason=attempt.get('grade_recovery_reason')))
            title = f"ThinkingCap offline grade {args.job} a{attempt['attempt']}"
            source = attempt['kernel']
        if api.kernels_status(source).status.name != 'COMPLETE':
            raise ValueError('Raw source output is not complete/attachable')
        # Keep only one campaign CPU grader active, independently of its two GPU jobs.
        for row in api.kernels_list(mine=True,page_size=100) or []:
            if any(s in row.ref for s in ('/thinkingcap-offline-grade-', '/thinkingcap-offline-cpu-verification', '/thinkingcap-offline-subset-reconcile')):
                if api.kernels_status(row.ref).status.name in ('RUNNING','QUEUED','CANCEL_REQUESTED'):
                    raise RuntimeError('Wait for active CPU grader: '+row.ref)
        kernel = 'krcalkrcal/'+title.lower().replace(' ','-')
        folder = state/'cpu-submissions'/(kernel.split('/')[-1]+'-'+args.commit[:8])
        folder.mkdir(parents=True,exist_ok=False)
        script = build_launcher(args.commit)
        (folder/'run.py').write_text(script)
        metadata = dict(id=kernel,title=title,code_file='run.py',language='python',kernel_type='script',
            is_private='true',enable_gpu='false',enable_internet='true',dataset_sources=[],
            competition_sources=[],kernel_sources=[source],model_sources=[])
        (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        if attempt is not None:
            attempt.update(grade_kernel=kernel,grade_state='submission_pending',grade_commit=args.commit)
            ledger_file.write_text(json.dumps(ledger,indent=2)+'\n')
        response = api.kernels_push(str(folder),timeout='3600')
        if attempt is not None:
            attempt.update(grade_state='submitted',grade_url=response.url)
            ledger_file.write_text(json.dumps(ledger,indent=2)+'\n')
        print(json.dumps(dict(kernel=kernel,url=response.url,version=response.version_number,gpu=False)))


if __name__ == '__main__':
    main()
