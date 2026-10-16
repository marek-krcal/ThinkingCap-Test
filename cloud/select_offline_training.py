"""Select a small task-aligned training subset without reading test questions.

This prepares data only; it never submits jobs or mutates the live campaign.
The rule targets arithmetic word problems, not individual benchmark content.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

COUNTS = dict(gsm8k=1400, skywork=400, commonsense_qa=100, arc_easy=100)
STORY = re.compile(r'\b(bought|buy|sold|sell|cost|price|dollars?|money|each|per|left|remaining|total|altogether|how many|how much|hours?|minutes?|days?|miles?|feet|apples?|books?|students?|candies|cookies?|workers?)\b', re.I)
ABSTRACT = re.compile(r'\\(?:log|sin|cos|tan|sqrt|frac)|\b(triangle|circle|polynomial|quadratic|theorem|prove|integer solutions|divisib|prime|modulo|sequence|probability|permutation|rectangle|geometry)\b', re.I)


def task_score(row):
    text = row['messages'][-1]['content'].split('\n\nPut the final answer')[0]
    words = len(text.split())
    story = len(set(x.lower() for x in STORY.findall(text)))
    abstract = len(ABSTRACT.findall(text))
    # Short natural-language arithmetic receives priority over abstract maths.
    # These are declared heuristic features, not measured transfer estimates.
    return 3*min(story, 8) - 8*abstract + (2 if 15 <= words <= 100 else 0)


def select(rows, counts=None):
    counts = COUNTS if counts is None else counts
    selected = []
    for source, count in counts.items():
        candidates = [r for r in rows if r['dataset'] == source]
        if len(candidates) < count:
            raise ValueError('Insufficient training rows: '+source)
        def key(row):
            tie = hashlib.sha256(('task-subset-v1/'+row['id']).encode()).hexdigest()
            score = task_score(row) if source in ('gsm8k', 'skywork') else 0
            return (-score, tie)
        selected.extend(sorted(candidates, key=key)[:count])
    if len({r['id'] for r in selected}) != len(selected):
        raise ValueError('Duplicate selected IDs')
    # Interleave sources in a stable order, avoiding long homogeneous runs.
    return sorted(selected, key=lambda r: hashlib.sha256(('subset-order/'+r['id']).encode()).hexdigest())


def prepare(source, out):
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(out)
    rows = [json.loads(line) for line in (source/'train.jsonl').read_text().splitlines()]
    if any(r.get('split') != 'train' for r in rows):
        raise ValueError('Only the frozen training split can be selected')
    selected = select(rows)
    out.mkdir(parents=True)
    (out/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    for name in ('validation.jsonl', 'test.jsonl'):
        shutil.copyfile(source/name, out/name)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = json.loads((source/'data-manifest.json').read_text())
    selection = dict(schema_version=1, method='task-subset-v1', counts=COUNTS,
        parent_train_sha256=sha(source/'train.jsonl'), train_sha256=sha(out/'train.jsonl'),
        selected_ids=[r['id'] for r in selected], test_content_used_for_selection=False,
        note='Heuristic arithmetic task alignment; no direct code/tool/constraint training source.')
    manifest['selection'] = {k:v for k,v in selection.items() if k!='selected_ids'}
    manifest['files']['train.jsonl'] = dict(prompts=len(selected), sha256=sha(out/'train.jsonl'),
        counts=dict(Counter(r['dataset'] for r in selected)))
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    (out/'data-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in selection.items() if k!='selected_ids'},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data/offline-input')
    parser.add_argument('--out',default='data/offline-subset')
    args=parser.parse_args()
    prepare(args.data,args.out)
