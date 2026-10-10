"""Apply the documented truncation rule to saved grades; never regenerate answers.

Preserves raw artifacts and generation metadata. Only truncated IFEval grades
are changed; all other verdicts remain the original graders' output.
Run as: python -m cloud.correct_truncation --source DIR --out NEW_DIR
"""
import argparse
import hashlib
import json
from pathlib import Path

from evaluate import collect, grade_one


def correct(source, out, questions):
    results_path = source / 'results.json'
    records_path = source / 'records.jsonl'
    results = json.loads(results_path.read_text())
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    cohort = [json.loads(line) for line in questions.read_text().splitlines()]
    if hashlib.sha256(questions.read_bytes()).hexdigest() != results['cohort_sha256']:
        raise ValueError('Cohort hash mismatch')
    if any(r.get('grading_error') or type(r.get('correct')) is not bool for r in records):
        raise ValueError('Cannot preserve failed or missing grades')
    by_id = {q['id']: q for q in cohort}
    changed, reviewed = [], []
    for record in records:
        if by_id[record['id']]['grader'] == 'ifeval' and record['truncated']:
            reviewed.append(record['id'])
            if record['correct']:
                changed.append(record['id'])
            correct_grade, error = grade_one(by_id[record['id']], record, {})
            record.update(correct=correct_grade, parse_error=error)
    results['results'], results['failed_benchmarks'] = collect(cohort, records, results['samples'])
    results['grading_correction'] = {
        'rule': 'Truncated IFEval answers count wrong, consistent with README and other graders',
        'source': str(source.resolve()),
        'source_results_sha256': hashlib.sha256(results_path.read_bytes()).hexdigest(),
        'source_records_sha256': hashlib.sha256(records_path.read_bytes()).hexdigest(),
        'reviewed_truncated_ifeval_ids': reviewed, 'correct_to_wrong_ids': changed,
        'regenerated_answers': 0,
    }
    # Refuse an existing destination rather than overwrite original artifacts.
    out.mkdir(parents=True, exist_ok=False)
    (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    (out / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    return results['grading_correction']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--questions', type=Path, default=Path('questions/questions.jsonl'))
    args = parser.parse_args()
    print(json.dumps(correct(args.source, args.out, args.questions), indent=2))
