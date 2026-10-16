"""Freeze all four README training sets with the original 200/source validation split."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.train import prepare_data as original

SOURCES = [
    ('gsm8k', original.gsm8k, 'openai/gsm8k', '740312add88f781978c0658806c59bc2815b9866'),
    ('skywork', original.skywork, 'Skywork/Skywork-OR1-RL-Data', '1cdedc52e0e2db85fdf252f9be682e63a5a38c33'),
    ('commonsense_qa', original.commonsense_qa, 'tau/commonsense_qa', '94630fe30dad47192a8546eb75f094926d47e155'),
    ('arc_easy', original.arc_easy, 'allenai/ai2_arc', '210d026faf9955653af8916fad021475a3f00453'),
]


def prepare(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    rng = random.Random(0)
    splits = {'train': [], 'validation': []}
    index, source_counts = 1000000, {}
    for source, loader, repo, revision in SOURCES:
        rows = []
        for ordinal, raw in enumerate(loader(revision)):
            content_hash = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
            row = dict(raw, id=f'{source}/{ordinal}', dataset=source,
                       seed_index=index, content_sha256=content_hash,
                       messages=raw['prompt'], dataset_repo=repo, dataset_revision=revision)
            rows.append(row)
            index += 1
        source_counts[source] = len(rows)
        rng.shuffle(rows)
        for i, row in enumerate(rows):
            split = 'validation' if i < 200 else 'train'
            row['split'] = split
            splits[split].append(row)
        print(source, len(rows), 'prompts', flush=True)
    rng.shuffle(splits['train'])
    manifest = dict(schema_version=1, split_seed=0, validation_per_source=200,
        sources={s: dict(repo=repo, revision=rev, prompts=source_counts[s])
                 for s, _, repo, rev in SOURCES}, files={})
    for split, rows in splits.items():
        path = out/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        manifest['files'][path.name] = dict(prompts=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              counts=dict(Counter(r['source'] for r in rows)))
    # Flag exact prepared duplicates for later training selection. Keep the full
    # requested source rows, never silently relabel or mix held-out splits.
    hashes = {}
    for split, rows in splits.items():
        for row in rows:
            hashes.setdefault(row['content_sha256'], set()).add(split)
    manifest['exact_content_hashes_crossing_train_validation'] = sum(len(v)>1 for v in hashes.values())
    (out/'data-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='data/offline-input')
    prepare(parser.parse_args().out)
