"""400-answer GPU generation, then two independent CPU-only saved-answer grades."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloud.run_eval import evaluation_command
from utils.offline import digest, grade_saved, read_generation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='/kaggle/working/runs/offline-verification')
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    args = SimpleNamespace(out=str(out/'raw'), model='Qwen/Qwen3-0.6B', adapter=None,
        samples=1, benchmarks='mbpp,bfcl,svamp,ifeval', limit_per_benchmark=100, smoke=False)
    subprocess.run(evaluation_command(args)+['--generate-only'], cwd=ROOT, check=True)
    generated = time.perf_counter()
    manifest, questions, records = read_generation(out/'raw')
    assert len(questions) == len(records) == 400
    raw_hash = digest(out/'raw/records.jsonl')
    first = grade_saved(out/'raw', out/'graded', str(ROOT/'questions'))
    second = grade_saved(out/'raw', out/'graded-again', str(ROOT/'questions'))
    verdicts = lambda rows: {(r['id'], r['sample_index']): r['correct'] for r in rows}
    assert verdicts(first) == verdicts(second), 'Repeated saved-answer grading changed verdicts'
    assert digest(out/'raw/records.jsonl') == raw_hash, 'Grading mutated raw trajectories'
    summary = dict(status='passed', answers=400, generated_answers=400,
        regenerated_for_grading=0, repeated_grading_identical=True,
        raw_records_sha256=raw_hash, generation=manifest['generation'],
        sampling=manifest['sampling'], caps=manifest['caps'],
        generation_stage_seconds=generated-started,
        grading_twice_seconds=time.perf_counter()-generated,
        total_seconds=time.perf_counter()-started)
    (out/'verification-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
