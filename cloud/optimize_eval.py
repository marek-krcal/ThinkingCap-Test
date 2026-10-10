"""Bounded sequential 400-question throughput campaign; no overlapping profiles."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from evaluate import BASE_MODEL, BASE_REVISION, BENCHMARKS, limit_questions_per_benchmark
from utils.eval.recipe import SAMPLING, SEED


def inspect_run(out, cohort):
    results = json.loads((out / 'results.json').read_text())
    records = [json.loads(line) for line in (out / 'records.jsonl').read_text().splitlines()]
    expected = {(q['id'], 0): q for q in cohort}
    keys = [(r['id'], r['sample_index']) for r in records]
    if len(keys) != 400 or len(set(keys)) != 400 or set(keys) != set(expected):
        raise ValueError('Profile is not exactly the requested 400 questions x1')
    if results['model'] != BASE_MODEL or results['revision'] != BASE_REVISION or results['dtype'] != 'half':
        raise ValueError('Model/revision/precision changed')
    if results['sampling'] != SAMPLING or results['seed'] != SEED or results['samples'] != 1:
        raise ValueError('Sampling recipe changed')
    if results['failed_benchmarks'] or any(type(r.get('correct')) is not bool or r.get('grading_error') for r in records):
        raise ValueError('Incomplete grading')
    for r in records:
        cap = 8192 if r['dataset'] == 'mbpp' else 4096
        if results['max_tokens'][r['dataset']] != cap or not 0 < r['output_tokens'] <= cap:
            raise ValueError('Token cap changed or invalid output token count')
        if r['seed'] != SEED + expected[(r['id'], 0)]['seed_index'] * 1000:
            raise ValueError('Answer seed changed')
        if r['truncated'] and r['correct']:
            raise ValueError('Truncated answer counted correct')
    scores = {}
    for benchmark, datasets in BENCHMARKS.items():
        subset = [r for r in records if r['dataset'] in datasets]
        scores[benchmark] = {'accuracy': sum(r['correct'] for r in subset)/len(subset),
            'questions': len(subset), 'tokens': sum(r['output_tokens'] for r in subset),
            'truncated': sum(r['truncated'] for r in subset)}
    event_paths = sorted((out / 'shards').glob('*/events.jsonl')) if results['data_parallel_size'] > 1 else [out / 'events.jsonl']
    starts, finishes = [], []
    for path in event_paths:
        events = [json.loads(line) for line in path.read_text().splitlines()]
        starts.append(next(e['time'] for e in events if e['phase'] == 'generation_started'))
        finishes.append(next(e['time'] for e in events if e['phase'] == 'generation_finished'))
    elapsed = max(finishes) - min(starts)
    total_tokens = sum(r['output_tokens'] for r in records)
    return {'verified': True, 'scores': scores, 'generation_seconds': elapsed,
        'generation_started': min(starts), 'generation_finished': max(finishes),
        'output_tokens': total_tokens, 'tokens_per_second': total_tokens/elapsed,
        'cohort_sha256': results['cohort_sha256'],
        'selected_ids_sha256': hashlib.sha256(json.dumps(sorted(str(q['id']) for q in cohort)).encode()).hexdigest()}


def clean_gpu():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    if any(line.strip().isdigit() for line in output.splitlines()):
        raise RuntimeError('GPU processes remain: refusing to overlap profiles')


def command(profile, out):
    args = [sys.executable, 'cloud/run_eval.py', '--out', str(out), '--backend', 'vllm',
        '--dtype', 'half', '--samples', '1', '--limit-per-benchmark', '100',
        '--tensor-parallel-size', '1', '--data-parallel-size', str(profile['dp']),
        '--batch-size', '128', '--max-num-seqs', str(profile['seqs'])]
    if profile.get('continuous'):
        args.append('--continuous-queue')
    return args


def choose(rows):
    valid = [r for r in rows if r.get('verified') and not r.get('quality_warnings')]
    if not valid:
        raise ValueError('No verified quality-compatible profile')
    return max(valid, key=lambda r: r['tokens_per_second'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.timeout_seconds
    cohort = [json.loads(line) for line in Path('questions/questions.jsonl').read_text().splitlines()]
    cohort = limit_questions_per_benchmark(cohort, list(BENCHMARKS), 100)
    rows = []
    def save(status='running'):
        (args.out / 'optimization-summary.json').write_text(json.dumps({'profiles': rows,
            'recipe': '400 frozen questions x1, FP16, fixed seeds; baseline queue chunks=128',
            'status': status}, indent=2))
    def execute(profile):
        clean_gpu()
        remaining = deadline - time.monotonic()
        if remaining < 180:
            raise TimeoutError('Not enough campaign time to start another engine')
        out = args.out / profile['name']
        row = dict(profile, verified=False)
        print('PROFILE START ' + json.dumps(profile), flush=True)
        process = None
        started = time.perf_counter()
        try:
            process = subprocess.Popen(command(profile, out), start_new_session=True)
            code = process.wait(timeout=min(1800, remaining))
            if code:
                raise subprocess.CalledProcessError(code, process.args)
            row.update(inspect_run(out, cohort))
            reference = next((r for r in rows if r.get('verified')), None)
            row['quality_warnings'] = []
            if reference:
                if row['cohort_sha256'] != reference['cohort_sha256'] or row['selected_ids_sha256'] != reference['selected_ids_sha256']:
                    raise ValueError('Profile cohorts differ')
                for benchmark, score in row['scores'].items():
                    base = reference['scores'][benchmark]
                    if abs(score['accuracy'] - base['accuracy']) > .100001:
                        row['quality_warnings'].append(benchmark + ': accuracy drift >10pp')
                    if not .7 <= score['tokens']/base['tokens'] <= 1.3:
                        row['quality_warnings'].append(benchmark + ': token drift >30%')
        except Exception as error:
            row['verified'] = False
            row['error'] = str(error)
            print('PROFILE FAILURE ' + profile['name'] + ': ' + str(error), flush=True)
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            row['evaluation_wall_seconds'] = time.perf_counter() - started
            rows.append(row)
            save()
        time.sleep(2)
        clean_gpu()
        print('PROFILE FINISH ' + json.dumps(row), flush=True)
        return row
    try:
        execute({'name': 'tp1-batched-32', 'dp': 1, 'seqs': 32})
        execute({'name': 'dp2-batched-32', 'dp': 2, 'seqs': 32})
        dp = choose(rows)['dp']
        execute({'name': f'dp{dp}-continuous-32', 'dp': dp, 'seqs': 32, 'continuous': True})
        execute({'name': f'dp{dp}-continuous-64', 'dp': dp, 'seqs': 64, 'continuous': True})
        if rows[-1].get('verified') and not rows[-1].get('quality_warnings'):
            execute({'name': f'dp{dp}-continuous-128', 'dp': dp, 'seqs': 128, 'continuous': True})
        winner = choose(rows)
        summary = {'status': 'complete', 'profiles': rows, 'winner': winner['name'],
            'selection_metric': 'output tokens per generation wall-second, quality-compatible profiles',
            'warning': 'Single trial per profile; rates depend on sampled lengths. Confirm before claiming a speedup.'}
        (args.out / 'optimization-summary.json').write_text(json.dumps(summary, indent=2))
        print('OPTIMIZATION COMPLETE ' + json.dumps(summary), flush=True)
    except BaseException:
        save('failed')
        raise


if __name__ == '__main__':
    main()
