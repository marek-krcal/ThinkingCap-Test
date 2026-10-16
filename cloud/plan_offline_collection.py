"""Freeze disjoint full-coverage shards; no generation or silent subsampling."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.offline import digest
from evaluate import BASE_MODEL, BASE_REVISION, load_suite
from utils.eval.recipe import SAMPLING, MAX_TOKENS, DEFAULT_MAX_TOKENS


def build_plan(root, questions, shard_size=400):
    root, questions = Path(root), Path(questions)
    manifest = json.loads((root/'data-manifest.json').read_text())
    suite = load_suite(questions)
    # Copy only the frozen question rows, not executable checkers; the latter
    # come from the pinned evaluation-cache input and are verified at grading.
    test = (questions/'questions.jsonl').read_bytes()
    (root/'test.jsonl').write_bytes(test)
    rows = [json.loads(line) for line in test.splitlines()]
    manifest['files']['test.jsonl'] = dict(prompts=len(rows), sha256=digest(root/'test.jsonl'),
        counts=dict(Counter(r['dataset'] for r in rows)))
    jobs = []
    for split in ('test', 'train', 'validation'):
        filename = split+'.jsonl'
        assert digest(root/filename) == manifest['files'][filename]['sha256']
        count = manifest['files'][filename]['prompts']
        for start in range(0, count, shard_size):
            jobs.append(dict(id=f'{split}-{start//shard_size:03d}', split=split,
                file=filename, start=start, stop=min(count, start+shard_size),
                prompts=min(shard_size, count-start),
                expected_answers=8*min(shard_size, count-start)))
    # Check exact prepared prompt overlap across held-out splits, in addition
    # to the original source IDs. Report it; do not silently drop source rows.
    import hashlib
    seen = {}
    for split in ('test', 'train', 'validation'):
        for line in (root/(split+'.jsonl')).read_text().splitlines():
            row = json.loads(line)
            content = json.dumps(row['messages'], sort_keys=True).encode()
            key = hashlib.sha256(content).hexdigest()
            seen.setdefault(key, set()).add(split)
    plan = dict(schema_version=1, samples=8, model=BASE_MODEL, revision=BASE_REVISION,
        sampling=SAMPLING, caps=MAX_TOKENS, default_cap=DEFAULT_MAX_TOKENS,
        data=manifest, benchmark_suite=suite, jobs=jobs,
        exact_message_hashes_crossing_splits=sum(len(v)>1 for v in seen.values()),
        expected_answers=sum(job['expected_answers'] for job in jobs))
    (root/'collection-plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(dict(jobs=len(jobs), expected_answers=plan['expected_answers'],
        files=manifest['files'], plan_sha256=digest(root/'collection-plan.json')), indent=2))
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/offline-input')
    parser.add_argument('--questions', default='questions')
    options = parser.parse_args()
    build_plan(options.data, options.questions)
