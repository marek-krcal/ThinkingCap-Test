"""Verify a downloaded full evaluation before declaring the campaign complete."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def compare_development_slice(records, cohort, reference, limit=100):
    """Compare the same deterministic slice, not differently weighted cohorts.

    v6 retained aggregate logs, not per-answer artifacts. These are diagnostic
    flags (10 percentage points / 30% token length), not paired significance
    tests, which would require the missing per-answer reference records.
    """
    groups = {'mbpp': ['mbpp'], 'bfcl': ['bfcl_simple_python', 'bfcl_multiple', 'bfcl_irrelevance'],
              'svamp': ['svamp'], 'ifeval': ['ifeval']}
    reports = {}
    for benchmark, datasets in groups.items():
        pools = {dataset: [q for q in cohort if q['dataset'] == dataset] for dataset in datasets}
        positions = Counter()
        chosen = []
        while len(chosen) < limit:
            added = False
            for dataset in datasets:
                if len(chosen) >= limit:
                    break
                if positions[dataset] < len(pools[dataset]):
                    chosen.append(pools[dataset][positions[dataset]]['id'])
                    positions[dataset] += 1
                    added = True
            if not added:
                break
        wanted = set(chosen)
        subset = [r for r in records if r['id'] in wanted and r['sample_index'] == 0]
        baseline = reference['benchmark_scores'][benchmark]
        expected_count = baseline['questions']
        baseline_accuracy = baseline['correct'] / expected_count
        if benchmark == 'bfcl':
            baseline_mean = sum(row['questions'] * row['mean_output_tokens']
                                for row in reference['bfcl_dataset_scores'].values()) / expected_count
        else:
            baseline_mean = baseline['mean_output_tokens']
        accuracy = sum(r.get('correct') is True for r in subset) / len(subset) if subset else 0
        mean = sum(r['output_tokens'] for r in subset) / len(subset) if subset else 0
        delta = 100 * (accuracy - baseline_accuracy)
        ratio = mean / baseline_mean
        warnings = []
        if len(subset) != expected_count or len(wanted) != expected_count:
            warnings.append('Incomplete matched reference slice')
        if abs(delta) > 10.000001:
            warnings.append('Accuracy differs by more than 10 percentage points: investigate')
        if not .7 <= ratio <= 1.3:
            warnings.append('Mean token length differs by more than 30%: investigate')
        reports[benchmark] = {'questions': len(subset), 'reference_accuracy': baseline_accuracy,
            'current_accuracy': accuracy, 'accuracy_change_pp': delta,
            'reference_mean_tokens': baseline_mean, 'current_mean_tokens': mean,
            'token_length_ratio': ratio, 'warnings': warnings}
    return reports


def verify(out, questions, expected_questions=2759, samples=1):
    results = json.loads((out / 'results.json').read_text())
    records = [json.loads(line) for line in (out / 'records.jsonl').read_text().splitlines()]
    raw = (questions / 'questions.jsonl').read_bytes()
    cohort = [json.loads(line) for line in raw.splitlines()]
    manifest = json.loads((questions / 'manifest.json').read_text())
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == manifest['cohort_sha256'] == results['cohort_sha256'], 'Cohort hash mismatch')
    require(len(cohort) == expected_questions == results['cohort_total'], 'Wrong cohort size')
    require(results['selection'] == 'full' and not results['smoke'], 'Not a full-cohort evaluation')
    require(results['samples'] == samples, 'Wrong number of samples')
    require(results['model'] == 'Qwen/Qwen3-0.6B' and
            results['revision'] == 'c1899de289a04d12100db370d81485cdf75e47ca' and
            results['adapter'] is None, 'Unexpected model, revision or adapter')
    require(results['sampling'] == {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20}
            and results['seed'] == 9001, 'Sampling recipe changed')
    require(results['revisions'] == manifest['revisions'], 'Benchmark revisions changed')
    by_id = {question['id']: question for question in cohort}
    require(len(by_id) == len(cohort), 'Duplicate question IDs in cohort')
    expected = {(question['id'], index) for question in cohort for index in range(samples)}
    keys = [(record['id'], record['sample_index']) for record in records]
    require(len(keys) == len(expected) and set(keys) == expected, 'Missing or duplicate answers')
    counts = Counter(question['dataset'] for question in cohort)
    require(dict(counts) == manifest['counts'], 'Manifest dataset counts mismatch')
    require(results['max_tokens'] == {dataset: 8192 if dataset == 'mbpp' else 4096
                                      for dataset in counts}, 'Token caps changed')
    for record in records:
        question = by_id.get(record['id'])
        if question is None:
            continue
        require(record['dataset'] == question['dataset'], 'Answer dataset mismatch')
        require(record['seed'] == 9001 + question['seed_index'] * 1000 + record['sample_index'],
                'Answer seed mismatch')
        require(type(record.get('correct')) is bool and not record.get('grading_error'),
                'Missing grade or grader error')
        require(not (record.get('truncated') and record.get('correct') is True),
                'Truncated answer counted correct')
        require(isinstance(record.get('text'), str) and bool(record['text']), 'Missing answer text')
        require(0 < record['output_tokens'] <= results['max_tokens'][record['dataset']],
                'Output token count exceeds cap')
    require(not results['failed_benchmarks'], 'Failed benchmark graders')
    require(set(results['results']) == set(counts), 'Missing dataset summaries')
    for dataset, count in counts.items():
        summary = results['results'].get(dataset, {})
        subset = [record for record in records if record['dataset'] == dataset]
        require(summary.get('n') == len(subset) == count * samples and
                summary.get('prompts') == count and summary.get('ungraded') == 0,
                f'Incomplete grading summary: {dataset}')
        accuracy = sum(record.get('correct') is True for record in subset) / len(subset) if subset else 0
        require(math.isclose(summary.get('accuracy', -1), accuracy), f'Score mismatch: {dataset}')
    generation = results.get('generation') or {}
    require(generation.get('responses') == len(records), 'Generation response count mismatch')
    require(generation.get('output_tokens') == sum(record['output_tokens'] for record in records),
            'Generation token total mismatch')
    return {'verified': not errors, 'errors': sorted(set(errors)), 'questions': len(cohort),
            'answers': len(records), 'samples': samples,
            'truncated': sum(bool(record.get('truncated')) for record in records),
            'grader_errors': sum(bool(record.get('grading_error')) for record in records)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--questions', type=Path, default=Path('questions'))
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--development-reference', type=Path,
                        help='Optional log-derived 400-question reference summary')
    args = parser.parse_args()
    report = verify(args.out, args.questions, samples=args.samples)
    if args.development_reference:
        records = [json.loads(line) for line in (args.out / 'records.jsonl').read_text().splitlines()]
        cohort = [json.loads(line) for line in (args.questions / 'questions.jsonl').read_text().splitlines()]
        reference = json.loads(args.development_reference.read_text())
        report['matched_400_comparison'] = compare_development_slice(records, cohort, reference)
        report['comparison_needs_investigation'] = any(
            item['warnings'] for item in report['matched_400_comparison'].values())
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if not report['verified'] else
                     2 if report.get('comparison_needs_investigation') else 0)
