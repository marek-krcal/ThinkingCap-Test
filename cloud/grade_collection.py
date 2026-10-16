"""CPU-only grading of a frozen raw generation artifact, without a model import."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.offline import digest, grade_saved, read_generation


def main():
    started = time.perf_counter()
    manifests = list(Path('/kaggle/input').rglob('generation-manifest.json'))
    if len(manifests) != 1:
        raise ValueError('Attach exactly one raw trajectory output to this CPU grader')
    source = manifests[0].parent
    manifest, questions, records = read_generation(source, allow_partial=True)
    out = Path('/kaggle/working/collection')
    out.mkdir(parents=True, exist_ok=False)
    raw_hash = digest(source/'records.jsonl')
    graded = grade_saved(source, out/'graded', str(ROOT/'questions'), allow_partial=True)
    if 'torch' in sys.modules or 'vllm' in sys.modules:
        raise RuntimeError('CPU grading unexpectedly imported a model runtime')
    if digest(source/'records.jsonl') != raw_hash:
        raise ValueError('CPU grading mutated raw source')
    reference = source.parent/'graded/records.jsonl'
    if reference.exists():
        original = [json.loads(line) for line in reference.read_text().splitlines()]
        verdicts = lambda rows: {(r['id'],r['sample_index']):r['correct'] for r in rows}
        if verdicts(original) != verdicts(graded):
            raise ValueError('CPU saved-answer grades differ from original GPU-session grades')
    fields = ('id', 'dataset', 'sample_index', 'seed', 'correct', 'output_tokens',
        'truncated', 'boundary_error', 'finish_reason', 'stop_reason')
    (out/'labels.jsonl').write_text(''.join(json.dumps({k:r.get(k) for k in fields})+'\n' for r in graded))
    prior_index = source.parent/'collection-index.json'
    if prior_index.exists():
        index = json.loads(prior_index.read_text())
        (out/'collection-index.json').write_text(json.dumps(index, indent=2)+'\n')
        summary = json.loads((source.parent/'collection-summary.json').read_text())
    else:
        summary = dict(kind='cpu_verification', answers=len(graded))
    summary.update(grading_status='completed', correct=sum(r['correct'] for r in graded),
        truncated=sum(r['truncated'] for r in graded), output_tokens=sum(r['output_tokens'] for r in graded),
        grading_seconds=time.perf_counter()-started, gpu_used=False, model_runtime_imported=False,
        raw_records_sha256=raw_hash)
    if reference.exists(): summary['cpu_verdicts_match_gpu'] = True
    (out/'collection-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='completed_keys'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
