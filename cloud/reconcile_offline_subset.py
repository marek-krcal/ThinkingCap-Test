"""CPU-only import of already generated selected trajectories; no generation.

Run on Kaggle with the new selected input and every completed old training
attempt attached. Outputs form immutable per-shard bootstrap parents.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.offline import digest, read_generation
from utils.train.rewards import is_task_correct


def reconcile(data, sources, out):
    data,out=Path(data),Path(out)
    if out.exists(): raise FileExistsError(out)
    plan=json.loads((data/'collection-plan.json').read_text())
    if plan.get('collection_splits') != ['train']:
        raise ValueError('Reconcile only the selected training plan')
    selected=[json.loads(l) for l in (data/'train.jsonl').read_text().splitlines()]
    if digest(data/'train.jsonl') != plan['data']['files']['train.jsonl']['sha256']:
        raise ValueError('Selected data changed')
    by_id={q['id']:q for q in selected}
    saved, prompts, provenance, template = {}, {}, [], None
    all_keys=set()
    for source in map(Path,sources):
        manifest,questions,records=read_generation(source,allow_partial=True)
        args=manifest['arguments']
        if (manifest['kind']!='train' or args['model']!=plan['model'] or
                args['revision']!=plan['revision'] or args['samples']!=8 or args['seed']!=9001 or
                args.get('dtype') not in ('half','float16') or
                manifest['sampling']!=plan['sampling'] or any(c!=4096 for c in manifest['caps'].values())):
            raise ValueError('Old generation recipe differs')
        template=args
        labels_path=source.parent/'labels.jsonl'
        labels=[json.loads(l) for l in labels_path.read_text().splitlines()]
        labels_by_key={(r['id'],r['sample_index']):r for r in labels}
        keys={(r['id'],r['sample_index']) for r in records}
        if len(labels_by_key)!=len(labels) or set(labels_by_key)!=keys or all_keys & keys:
            raise ValueError('Duplicate or missing old answer/label keys')
        all_keys.update(keys)
        source_questions={q['id']:q for q in questions}
        for record in records:
            key=(record['id'],record['sample_index'])
            if record['id'] not in by_id: continue
            original=source_questions[record['id']]; desired=by_id[record['id']]
            for field in ('id','dataset','seed_index','messages','answer','task','split','dataset_revision'):
                if original.get(field)!=desired.get(field):
                    raise ValueError('Selected question identity changed: '+field)
            label=labels_by_key[key]
            correct=bool(not record['truncated'] and not record.get('boundary_error') and
                         is_task_correct(record['text'],desired['answer'],desired['task']))
            if type(label.get('correct')) is not bool or label['correct']!=correct:
                raise ValueError('Reused correctness differs from original grader')
            for field in ('seed','dataset','output_tokens','truncated','boundary_error','finish_reason','stop_reason'):
                if label.get(field)!=record.get(field): raise ValueError('Reused label mismatch: '+field)
            saved[key]=(record,label)
            prompts[record['id']]=original
        provenance.append(dict(source=str(source),manifest_sha256=digest(source/'generation-manifest.json'),
            records_sha256=digest(source/'records.jsonl'),prompts_sha256=digest(source/'prompts.jsonl'),
            labels_sha256=digest(labels_path),answers=len(records)))
    if template is None: raise ValueError('Attach all completed old training attempts')
    out.mkdir(parents=True)
    for job in plan['jobs']:
        questions=selected[job['start']:job['stop']]
        allowed={(q['id'],i) for q in questions for i in range(8)}
        keys=sorted(set(saved)&allowed)
        folder=out/'bootstrap'/job['id']; raw=folder/'raw'; raw.mkdir(parents=True)
        active=sorted({key[0] for key in keys})
        raw_questions=[prompts[id] for id in active]
        (raw/'prompts.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in raw_questions))
        (raw/'records.jsonl').write_text(''.join(json.dumps(saved[key][0])+'\n' for key in keys))
        args=dict(template,resume_keys=sorted({(q['id'],i) for q in raw_questions for i in range(8)}-set(keys)))
        manifest=dict(schema_version=1,kind='train',status='completed',arguments=args,suite={},
            caps={q['dataset']:4096 for q in raw_questions},sampling=plan['sampling'],
            prompts_sha256=digest(raw/'prompts.jsonl'),records_sha256=digest(raw/'records.jsonl'),
            expected_answers=len(keys),completed_answers=len(keys),generation=None)
        (raw/'generation-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        read_generation(raw)
        labels=[saved[key][1] for key in keys]
        (folder/'labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
        index=dict(plan_sha256=digest(data/'collection-plan.json'),job_id=job['id'],split='train',
            expected_answers=job['expected_answers'],completed_keys=keys,new_answers=len(keys),
            cumulative_answers=len(keys),status='completed' if len(keys)==job['expected_answers'] else 'partial',
            raw_records_sha256=digest(raw/'records.jsonl'),reused=True)
        (folder/'collection-index.json').write_text(json.dumps(index,indent=2)+'\n')
        summary=dict(index,grading_status='completed',generation=None,correct=sum(r['correct'] for r in labels),
            truncated=sum(r['truncated'] for r in labels),output_tokens=sum(r['output_tokens'] for r in labels))
        (folder/'collection-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=dict(plan_sha256=digest(data/'collection-plan.json'),train_sha256=digest(data/'train.jsonl'),
        reused_answers=len(saved),source_answers=len(all_keys),sources=provenance,
        regenerated_answers=0,gpu_used=False,model_runtime_imported=False)
    if 'torch' in sys.modules or 'vllm' in sys.modules: raise RuntimeError('Unexpected model runtime')
    (out/'reuse-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True)
    parser.add_argument('--source',action='append',required=True)
    parser.add_argument('--out',default='/kaggle/working/reconciled')
    args=parser.parse_args()
    print(json.dumps(reconcile(args.data,args.source,args.out),indent=2))
