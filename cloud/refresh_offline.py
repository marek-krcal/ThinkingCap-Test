"""Refresh the collection ledger using small metadata only; never fetch weights/raw corpora."""
import fcntl
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloud.collect_offline import load_job, prior_keys
from utils.offline import digest


def refresh(api, data, state):
    ledger_file = state/'ledger.json'
    ledger = json.loads(ledger_file.read_text())
    plan_hash = digest(data/'collection-plan.json')
    if ledger['plan_sha256'] != plan_hash:
        raise ValueError('Ledger plan changed')
    for attempt in ledger['attempts']:
        if attempt['state'] in ('completed', 'partial', 'cancelled_startup'):
            continue
        status = api.kernels_status(attempt['kernel']).status.name
        attempt['remote_status'] = status
        if status in ('RUNNING', 'QUEUED', 'CANCEL_REQUESTED'):
            continue
        destination = state/'artifacts'/attempt['kernel'].split('/')[-1]
        # The current Kaggle SDK automatically follows all output page tokens.
        api.kernels_output(attempt['kernel'], path=str(destination), page_size=200,
            file_pattern=r'(collection-(summary|index)\.json|labels\.jsonl|generation-manifest\.json|results\.json|grading-provenance\.json|setup-timing\.json|run-config\.json)$', quiet=True)
        attempt['artifacts'] = str(destination.relative_to(ROOT))
        if status != 'COMPLETE':
            attempt['state'] = 'failed'
            continue
        index_path = destination/'collection/collection-index.json'
        summary_path = destination/'collection/collection-summary.json'
        if not index_path.exists() or not summary_path.exists():
            attempt['state'] = 'audit_required'
            continue
        plan, job, questions = load_job(data, attempt['job_id'])
        keys = prior_keys(index_path, plan_hash, job, questions)
        summary = json.loads(summary_path.read_text())
        index = json.loads(index_path.read_text())
        attempt.update(completed_answers=len(keys), new_answers=index['new_answers'])
        if summary['grading_status'] == 'deferred_cpu':
            attempt['state'] = 'generated'
            if not attempt.get('grade_kernel'):
                continue
            grade_status = api.kernels_status(attempt['grade_kernel']).status.name
            attempt['grade_remote_status'] = grade_status
            if grade_status in ('RUNNING','QUEUED','CANCEL_REQUESTED'):
                continue
            if grade_status != 'COMPLETE':
                attempt['state'] = 'audit_required'
                continue
            # Overlay small CPU grading artifacts, never raw trajectories.
            api.kernels_output(attempt['grade_kernel'], path=str(destination),page_size=200,
                file_pattern=r'(collection-(summary|index)\.json|labels\.jsonl|results\.json|grading-provenance\.json)$',quiet=True,force=True)
            summary = json.loads(summary_path.read_text())
            index = json.loads(index_path.read_text())
        labels = [json.loads(line) for line in (destination/'collection/labels.jsonl').read_text().splitlines()]
        if (len(labels) != index['new_answers'] or any(type(r.get('correct')) is not bool for r in labels) or
                len({(r['id'],r['sample_index']) for r in labels}) != len(labels) or
                not {(r['id'],r['sample_index']) for r in labels}.issubset(set(keys)) or
                any(r['truncated'] and r['correct'] for r in labels)):
            raise ValueError('Incomplete or invalid grading labels')
        if (summary['grading_status'] != 'completed' or summary['completed_keys'] != index['completed_keys'] or
                index['cumulative_answers'] != len(keys) or index['expected_answers'] != job['expected_answers']):
            raise ValueError('Collection output coverage/grading audit failed')
        attempt.update(state=index['status'], completed_answers=len(keys), new_answers=index['new_answers'],
            output_tokens=summary['output_tokens'], correct=summary['correct'], truncated=summary['truncated'])
        if attempt['state'] == 'completed' and len(keys) != job['expected_answers']:
            raise ValueError('Shard falsely declared complete')
        ledger_file.write_text(json.dumps(ledger, indent=2)+'\n')
    latest = {a['job_id']: a for a in ledger['attempts']}
    plan = json.loads((data/'collection-plan.json').read_text())
    ledger['progress'] = dict(target_answers=plan['expected_answers'],
        saved_answers=sum(a.get('completed_answers',0) for a in latest.values()),
        completed_shards=sum(a['state']=='completed' for a in latest.values()),
        total_shards=len(plan['jobs']),
        pending_shards=[j['id'] for j in plan['jobs'] if j['id'] not in latest],
        incomplete_shards=[k for k,a in latest.items() if a['state'] in ('partial','cancelled_startup')],
        attention=[a['kernel'] for a in latest.values() if a['state'] in ('failed','audit_required')])
    # Aggregate small labels across every successful attempt, not just the
    # latest partial's new answers. A duplicated trajectory is a hard failure.
    by_key, by_split = {}, {}
    for attempt in ledger['attempts']:
        if attempt['state'] not in ('completed','partial'): continue
        labels_file = ROOT/attempt['artifacts']/'collection/labels.jsonl'
        for line in labels_file.read_text().splitlines():
            row = json.loads(line)
            key = (row['id'],row['sample_index'])
            if key in by_key: raise ValueError('Duplicate trajectory across successful attempts')
            by_key[key] = row
            split = attempt['job_id'].rsplit('-',1)[0]
            counters = by_split.setdefault(split, Counter())
            counters.update(answers=1, correct=int(row['correct']), truncated=int(row['truncated']),
                output_tokens=row['output_tokens'])
    ungraded = sum(a.get('new_answers',0) for a in ledger['attempts'] if a['state'] not in ('completed','partial'))
    if len(by_key)+ungraded != ledger['progress']['saved_answers']:
        raise ValueError('Cumulative indexes disagree with graded label coverage')
    ledger['progress']['graded_answers'] = len(by_key)
    ledger['progress']['awaiting_cpu_grading'] = [a['job_id'] for a in latest.values() if a['state']=='generated']
    ledger['progress']['partitions'] = {k:dict(v) for k,v in by_split.items()}
    if ledger['progress']['completed_shards'] == len(plan['jobs']):
        expected = set()
        for job in plan['jobs']:
            _, _, questions = load_job(data, job['id'])
            expected.update((row['id'], i) for row in questions for i in range(8))
        if set(by_key) != expected:
            raise ValueError('Full corpus does not match frozen requested coverage')
        ledger['progress']['full_coverage_verified'] = True
    ledger_file.write_text(json.dumps(ledger, indent=2)+'\n')
    print(json.dumps(ledger['progress'], indent=2))
    return ledger


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data/offline-input')
    parser.add_argument('--state',default='runs/offline-campaign')
    options=parser.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    state = ROOT/options.state
    with (state/'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refresh(api, ROOT/options.data, state)
