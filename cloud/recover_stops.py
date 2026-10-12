"""CPU-only audit of completed legacy vLLM outputs; never generate answers.

Only use after verifying a successful FINAL_ONLY run with no aborts, stop
strings or time-based generation stops, and prompt budget + output cap within
model context. The old artifact omitted finish_reason, so recovered stops are
explicitly inferred, not observed. Exact-cap answers remain conservative.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from evaluate import collect, grade_all, grading_context, load_suite
from utils.eval.graders import mbpp_code


def recover(records, results):
    if results['backend'] != 'vllm' or results['generation']['responses'] != len(records):
        raise ValueError('Expected a completed vLLM generation')
    if len({(r['id'], r['sample_index']) for r in records}) != len(records):
        raise ValueError('Duplicate answers')
    if results['max_model_len'] < 4096 + max(results['max_tokens'].values()):
        raise ValueError('Context could terminate generation before the output cap')
    changed = []
    for record in records:
        cap = results['max_tokens'][record['dataset']]
        if not 0 < record['output_tokens'] <= cap:
            raise ValueError('Invalid token count')
        if 'finish_reason' in record:
            raise ValueError('Use recorded finish_reason, not this legacy recovery')
        if record['truncated'] and record['output_tokens'] < cap:
            changed.append(record['id'])
            record.update(truncated=False,
                          completion_verified_by='inferred_completed_final_only_under_cap')
        for key in ('correct', 'parse_error', 'grading_error'):
            record.pop(key, None)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--completed-final-only-no-abort', action='store_true', required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result_path, record_path = args.source/'results.json', args.source/'records.jsonl'
    results = json.loads(result_path.read_text())
    records = [json.loads(line) for line in record_path.read_text().splitlines()]
    suite = load_suite(Path('questions'))
    if suite['cohort_sha256'] != results['cohort_sha256'] or suite['revisions'] != results['revisions']:
        raise ValueError('Cohort or grader revisions differ')
    questions = [json.loads(line) for line in Path('questions/questions.jsonl').read_text().splitlines()]
    by_id = {q['id']: q for q in questions}
    selected = [by_id[r['id']] for r in records if r['sample_index'] == 0]
    changed = recover(records, results)
    grading = grading_context(suite, SimpleNamespace(benchmarks=results['benchmarks'], sandbox=None, workers=4))
    # On a personal laptop, do not run generated programs without isolation.
    # SimPO's malformed/empty program answers all reject before execution.
    if grading['mode'] != 'bwrap':
        for record in records:
            row = by_id[record['id']]
            if row['grader'] == 'mbpp':
                _, error = mbpp_code(record['response'], row['entry_point'],
                                     grading['checkers']['mbpp'], record['truncated'])
                if not error:
                    raise RuntimeError('Recovered program needs an isolated grader; refusing local execution')
    grade_all(selected, records, grading)
    results['results'], results['failed_benchmarks'] = collect(selected, records, results['samples'])
    results['completion_correction'] = {
        'source': str(args.source.resolve()),
        'source_results_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
        'source_records_sha256': hashlib.sha256(record_path.read_bytes()).hexdigest(),
        'inferred_under_cap_completed_ids': changed,
        'assumption': 'successful FINAL_ONLY run, no abort/stop-string/time stop, context exceeds all budgets',
        'exact_cap_policy': 'retain original truncation flag',
        'regenerated_answers': 0,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    (args.out/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    print(json.dumps({'changed_completion_flags': len(changed),
                      'correct': sum(r['correct'] is True for r in records),
                      'grading_errors': sum(bool(r.get('grading_error')) for r in records)}))


if __name__ == '__main__':
    main()
