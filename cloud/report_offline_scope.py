"""Report only the revised selected-train + original test/validation target."""
import json
from collections import Counter
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]


def report(root=ROOT):
    root=Path(root)
    desired={}
    for split in ('train','test','validation'):
        data='offline-subset' if split=='train' else 'offline-input'
        for line in (root/'data'/data/(split+'.jsonl')).read_text().splitlines():
            row=json.loads(line)
            for sample in range(8):
                key=(row['id'],sample)
                if key in desired: raise ValueError('Target splits overlap')
                desired[key]=(split,row)
    saved=set();graded={};active=[]
    for state,selected in (('offline-campaign',False),('offline-subset-campaign',True)):
        path=root/'runs'/state/'ledger.json'
        if not path.exists(): continue
        ledger=json.loads(path.read_text())
        for attempt in ledger['attempts']:
            split=attempt['job_id'].rsplit('-',1)[0]
            if (split=='train')!=selected: continue
            if attempt['state'] in ('submitted','submission_pending'):
                active.append(attempt['kernel'])
            if not attempt.get('artifacts'): continue
            folder=root/attempt['artifacts']/'collection'
            index=folder/'collection-index.json'
            if not index.exists(): continue
            keys={tuple(k) for k in json.loads(index.read_text())['completed_keys']}
            if not keys.issubset(desired): raise ValueError('Saved keys outside revised target')
            saved.update(keys)
            if attempt['state'] not in ('partial','completed'): continue
            for line in (folder/'labels.jsonl').read_text().splitlines():
                row=json.loads(line);key=(row['id'],row['sample_index'])
                if key in graded: raise ValueError('Duplicated graded key in revised target')
                if key not in desired: raise ValueError('Unexpected graded answer')
                _,question=desired[key]
                if row['seed']!=9001+question['seed_index']*1000+row['sample_index']:
                    raise ValueError('Revised answer seed changed')
                if type(row['correct']) is not bool or (row['truncated'] and row['correct']):
                    raise ValueError('Invalid revised correctness')
                graded[key]=row
    if not set(graded).issubset(saved): raise ValueError('Graded keys not saved')
    metrics={}
    for key,row in graded.items():
        split,_=desired[key]
        metric=metrics.setdefault(split,Counter())
        metric.update(answers=1,correct=int(row['correct']),truncated=int(row['truncated']),output_tokens=row['output_tokens'])
    result=dict(checked_at=time.time(),target_answers=len(desired),saved_answers=len(saved),
        graded_answers=len(graded),awaiting_grading=len(saved)-len(graded),
        partitions={k:dict(v) for k,v in metrics.items()},active_collectors=sorted(set(active)),
        full_coverage_verified=saved==set(desired) and set(graded)==set(desired))
    destination=root/'runs/offline-subset-campaign/scope-progress.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':print(json.dumps(report(),indent=2))
