"""Submit/audit the one CPU-only import before selected-training generation."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.offline import digest
from cloud.collect_offline import load_job, prior_keys

KERNEL='krcalkrcal/thinkingcap-offline-subset-reconcile'
DATASET='krcalkrcal/thinkingcap-offline-subset-input'


def audit(api,data,state):
    ledger_path=state/'ledger.json';ledger=json.loads(ledger_path.read_text())
    if ledger.get('reuse_pending') is False: raise ValueError('Already imported; never import twice')
    if api.kernels_status(KERNEL).status.name!='COMPLETE': raise RuntimeError('Reconciliation not COMPLETE')
    destination=state/'artifacts'/KERNEL.split('/')[-1]
    api.kernels_output(KERNEL,path=str(destination),page_size=200,quiet=True,
        file_pattern=r'(reuse-provenance\.json|collection-(summary|index)\.json|labels\.jsonl|generation-manifest\.json)$')
    base=destination/'reconciled'
    report=json.loads((base/'reuse-provenance.json').read_text())
    plan=json.loads((data/'collection-plan.json').read_text())
    if (report['plan_sha256']!=ledger['plan_sha256'] or report['regenerated_answers']!=0 or
            report['gpu_used'] or report['model_runtime_imported']):
        raise ValueError('Invalid reconciliation provenance')
    required=ledger['reuse_source_hashes']
    actual={s['records_sha256'] for s in report['sources']}
    if actual!=set(required.values()): raise ValueError('Missing or unexpected old raw sources')
    attempts=[];seen=set()
    for job in plan['jobs']:
        folder=base/'bootstrap'/job['id']
        _,_,questions=load_job(data,job['id'])
        keys=prior_keys(folder/'collection-index.json',ledger['plan_sha256'],job,questions)
        labels=[json.loads(l) for l in (folder/'labels.jsonl').read_text().splitlines()]
        summary=json.loads((folder/'collection-summary.json').read_text())
        label_keys={(r['id'],r['sample_index']) for r in labels}
        if (label_keys!=set(keys) or len(labels)!=len(keys) or seen&label_keys or
                any(type(r.get('correct')) is not bool or (r['truncated'] and r['correct']) for r in labels) or
                summary['grading_status']!='completed' or summary['new_answers']!=len(keys)):
            raise ValueError('Invalid imported labels')
        seen.update(label_keys)
        # refresh_offline expects artifacts/<attempt>/collection. Store only
        # small metadata locally; raw reused completions remain on Kaggle.
        local=state/'bootstrap'/job['id']/'collection';local.mkdir(parents=True,exist_ok=True)
        for name in ('collection-index.json','collection-summary.json','labels.jsonl'):
            (local/name).write_bytes((folder/name).read_bytes())
        attempts.append(dict(job_id=job['id'],attempt=1,kernel=KERNEL,state=summary['status'],
            remote_status='COMPLETE',timeout=0,reused=True,completed_answers=len(keys),new_answers=len(keys),
            artifacts=str(local.parent.relative_to(ROOT)),correct=summary['correct'],
            truncated=summary['truncated'],output_tokens=summary['output_tokens']))
    if len(seen)!=report['reused_answers']: raise ValueError('Reused coverage mismatch')
    ledger.update(reuse_pending=False,reconcile_state='audited',attempts=attempts)
    ledger_path.write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(dict(reused_answers=len(seen),gpu_used=False,regenerated_answers=0)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data/offline-subset')
    parser.add_argument('--state',default='runs/offline-subset-campaign')
    parser.add_argument('--commit')
    parser.add_argument('--audit',action='store_true')
    args=parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    data=ROOT/args.data;state=ROOT/args.state;state.mkdir(parents=True,exist_ok=True)
    with (state/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.audit: return audit(api,data,state)
        if not args.commit or not re.fullmatch('[0-9a-f]{40}',args.commit): raise ValueError('Full published SHA required')
        path=state/'ledger.json'
        if path.exists(): raise RuntimeError('Existing intent: inspect exact reconciliation session; never blindly resubmit')
        old=json.loads((ROOT/'runs/offline-campaign/ledger.json').read_text())
        sources=[a for a in old['attempts'] if a['job_id'].startswith('train-')]
        if not sources: raise ValueError('No old sources to reconcile')
        for a in sources:
            if a['state'] not in ('partial','completed') or api.kernels_status(a['kernel']).status.name!='COMPLETE':
                raise RuntimeError('Wait for all old training attempts to stop and be graded')
        for row in api.kernels_list(mine=True,page_size=100) or []:
            if any(s in row.ref for s in ('/thinkingcap-offline-grade-','/thinkingcap-offline-cpu-verification','/thinkingcap-offline-subset-reconcile')):
                if api.kernels_status(row.ref).status.name in ('RUNNING','QUEUED','CANCEL_REQUESTED'):
                    raise RuntimeError('Campaign CPU slot occupied: '+row.ref)
        folder=state/'reconcile-submission';folder.mkdir(exist_ok=False)
        env=dict(THINKINGCAP_REF=args.commit,THINKINGCAP_OFFLINE_DATASET=DATASET,
            THINKINGCAP_PLAN_SHA=digest(data/'collection-plan.json'),
            THINKINGCAP_REUSE_SOURCES=json.dumps([a['kernel'] for a in sources]))
        source=('import os,urllib.request\n'+f'os.environ.update({env!r})\n'+
            f'url="https://raw.githubusercontent.com/marek-krcal/ThinkingCap-Test/{args.commit}/cloud/kaggle-subset-reconcile.py"\n'+
            'with urllib.request.urlopen(url,timeout=60) as r: code=r.read().decode()\n'+
            'exec(compile(code,url,"exec"),{"__name__":"__main__","__file__":url})\n')
        (folder/'run.py').write_text(source)
        metadata=dict(id=KERNEL,title='ThinkingCap offline subset reconcile',code_file='run.py',language='python',
            kernel_type='script',is_private='true',enable_gpu='false',enable_internet='true',
            dataset_sources=[DATASET],kernel_sources=[a['kernel'] for a in sources],competition_sources=[],model_sources=[])
        (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        hashes={a['kernel']:json.loads((ROOT/a['artifacts']/'collection/collection-index.json').read_text())['raw_records_sha256'] for a in sources}
        ledger=dict(plan_sha256=digest(data/'collection-plan.json'),reuse_pending=True,
            reconcile_kernel=KERNEL,reconcile_commit=args.commit,reconcile_state='submission_pending',
            reuse_source_hashes=hashes,attempts=[])
        path.write_text(json.dumps(ledger,indent=2)+'\n')
        response=api.kernels_push(str(folder),timeout='900')
        ledger.update(reconcile_state='submitted',reconcile_version=response.version_number)
        path.write_text(json.dumps(ledger,indent=2)+'\n')
        print(json.dumps(dict(kernel=KERNEL,url=response.url,version=response.version_number,gpu=False)))


if __name__=='__main__':main()
