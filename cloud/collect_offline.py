"""A resumable, time-bounded two-T4 shard; grade only its saved complete answers."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import generate, load_suite, parse_arguments, prepare_output
from utils.offline import digest, grade_saved, read_generation, save_generation
from utils.eval.recipe import MAX_TOKENS, DEFAULT_MAX_TOKENS


def load_job(data, job_id):
    plan = json.loads((data/'collection-plan.json').read_text())
    if plan['samples'] != 8 or plan['caps'] != MAX_TOKENS or plan['default_cap'] != DEFAULT_MAX_TOKENS:
        raise ValueError('Sample count or token caps differ from frozen recipe')
    job = next(j for j in plan['jobs'] if j['id'] == job_id)
    source = data/job['file']
    if digest(source) != plan['data']['files'][job['file']]['sha256']:
        raise ValueError('Prepared source file changed')
    rows = [json.loads(line) for line in source.read_text().splitlines()][job['start']:job['stop']]
    if len(rows) != job['prompts'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Shard source coverage invalid')
    for row in rows:
        row['split'] = job['split']
    return plan, job, rows


def prior_keys(path, plan_hash, job, questions):
    if path is None:
        return []
    previous = json.loads(Path(path).read_text())
    if previous['plan_sha256'] != plan_hash or previous['job_id'] != job['id']:
        raise ValueError('Resume index belongs to a different shard or frozen recipe')
    keys = [tuple(k) for k in previous['completed_keys']]
    allowed = {(q['id'], i) for q in questions for i in range(8)}
    if len(keys) != len(set(keys)) or not set(keys).issubset(allowed):
        raise ValueError('Invalid resume coverage')
    return keys


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data', required=True)
    cli.add_argument('--job', required=True)
    cli.add_argument('--resume-index')
    cli.add_argument('--expected-plan-sha', required=True)
    cli.add_argument('--out', default='/kaggle/working/collection')
    cli.add_argument('--generation-seconds', type=int, default=2700)
    options = cli.parse_args()
    if not 60 <= options.generation_seconds <= 7200:
        cli.error('Generation safety budget must be 60..7200 seconds')
    data, out = Path(options.data), Path(options.out)
    plan, job, questions = load_job(data, options.job)
    plan_hash = digest(data/'collection-plan.json')
    if plan_hash != options.expected_plan_sha:
        raise ValueError('Collection plan differs from pinned dispatch input')
    previous = prior_keys(options.resume_index, plan_hash, job, questions)
    args = parse_arguments(['--out', str(out/'raw'), '--generate-only', '--samples', '8',
        '--backend', 'vllm', '--dtype', 'half', '--tensor-parallel-size', '1',
        '--data-parallel-size', '2', '--continuous-queue', '--max-num-seqs', '64',
        '--max-num-batched-tokens', '2048', '--gpu-memory-utilization', '0.70'])
    if args.model != plan['model'] or args.revision != plan['revision']:
        raise ValueError('Model recipe changed')
    args.sampling = plan['sampling']
    args.resume_keys = previous
    args.allow_partial = True
    args.shard_root = '/kaggle/temp/offline-worker-shards'
    # Deadline is shared by both spawned engines, including their cold startup.
    args.queue_deadline = time.time()+options.generation_seconds
    suite = dict(plan['benchmark_suite'], collection_shard=job['id']) if job['split']=='test' else {}
    kind = 'benchmark' if job['split']=='test' else job['split']
    if kind == 'benchmark':
        fresh = load_suite(ROOT/'questions')
        if fresh['cohort_sha256'] != suite['cohort_sha256'] or fresh['revisions'] != suite['revisions']:
            raise ValueError('Benchmark cache differs from frozen collection plan')
    prepare_output(out/'raw', False)
    save_generation(out/'raw', args, suite, questions, None, kind=kind)
    if len(previous) != job['expected_answers']:
        if job['expected_answers']-len(previous) == 1:
            args.data_parallel_size = 1  # final single missing trajectory only
        records = generate(questions, args, out/'raw')
    else:
        records = []
        (out/'raw/records.jsonl').write_text('')
    manifest = save_generation(out/'raw', args, suite, questions, records, kind=kind)
    read_generation(out/'raw', allow_partial=True)
    keys = sorted(set(previous)|{(r['id'], r['sample_index']) for r in records})
    index = dict(plan_sha256=plan_hash, job_id=job['id'], split=job['split'],
        expected_answers=job['expected_answers'], completed_keys=keys,
        new_answers=len(records), cumulative_answers=len(keys),
        status='completed' if len(keys)==job['expected_answers'] else 'partial',
        raw_records_sha256=manifest['records_sha256'])
    if options.resume_index:
        index['parent_index_sha256'] = digest(options.resume_index)
        index['parent_index_path'] = options.resume_index
    # Seal coverage before CPU grading: a grader failure must not invalidate
    # already generated, reusable trajectories or require generating them twice.
    (out/'collection-index.json').write_text(json.dumps(index, indent=2)+'\n')
    summary = dict(index, generation=manifest['generation'], grading_status='pending')
    (out/'collection-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    graded = grade_saved(out/'raw', out/'graded', str(ROOT/'questions'), allow_partial=True)
    # Lightweight labels let local audits aggregate correctness and cap rates
    # without downloading the potentially multi-GB text/token corpus.
    fields = ('id', 'dataset', 'sample_index', 'seed', 'correct', 'output_tokens',
        'truncated', 'boundary_error', 'finish_reason', 'stop_reason')
    (out/'labels.jsonl').write_text(''.join(json.dumps({k:r.get(k) for k in fields})+'\n' for r in graded))
    summary.update(grading_status='completed', correct=sum(r['correct'] for r in graded),
        truncated=sum(r['truncated'] for r in graded),
        output_tokens=sum(r['output_tokens'] for r in graded))
    (out/'collection-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='completed_keys'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
