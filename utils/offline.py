"""Immutable generation artifacts, provenance checks and separate CPU grading."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def requested_keys(questions, arguments):
    skipped = {tuple(key) for key in getattr(arguments, 'resume_keys', [])}
    return {(q['id'], i) for q in questions for i in range(arguments.samples)}-skipped


def save_generation(out, arguments, suite, questions, records, kind='benchmark'):
    from evaluate import caps_in_use, sampling_parameters
    out = Path(out)
    question_file = out/'prompts.jsonl'
    if not question_file.exists():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(arguments.model, revision=arguments.revision)
        with question_file.open('w') as stream:
            for question in questions:
                tokens = tokenizer.apply_chat_template(question['messages'], tokenize=True,
                      add_generation_prompt=True, enable_thinking=True)
                stream.write(json.dumps(dict(question, prompt_tokens=tokens))+'\n')
    manifest = dict(schema_version=1, kind=kind,
        status='generating' if records is None else 'completed',
        arguments=vars(arguments), suite=suite, caps=caps_in_use(questions, arguments),
        sampling=sampling_parameters(arguments), prompts_sha256=digest(question_file),
        expected_answers=len(requested_keys(questions, arguments)),
        generation=getattr(arguments, 'generation_stats', None))
    if records is not None:
        expected = requested_keys(questions, arguments)
        keys = [(r['id'], r['sample_index']) for r in records]
        if (len(keys) != len(set(keys)) or not set(keys).issubset(expected) or
                (not getattr(arguments, 'allow_partial', False) and set(keys) != expected)):
            raise ValueError('Generation missing or duplicating requested trajectories')
        if set(keys) != expected:
            manifest['status'] = 'partial'
        manifest['records_sha256'] = digest(out/'records.jsonl')
        manifest['completed_answers'] = len(records)
        with (out/'completed-keys.jsonl').open('w') as stream:
            for record in records:
                stream.write(json.dumps({k: record[k] for k in ('id', 'sample_index', 'seed')})+'\n')
    (out/'generation-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def read_generation(source, allow_partial=False):
    source = Path(source)
    manifest = json.loads((source/'generation-manifest.json').read_text())
    if manifest['status'] != 'completed' and not (allow_partial and manifest['status'] == 'partial'):
        raise ValueError('Generation is incomplete; do not report a full evaluation')
    for name, field in (('prompts.jsonl', 'prompts_sha256'), ('records.jsonl', 'records_sha256')):
        if digest(source/name) != manifest[field]:
            raise ValueError(f'Generation artifact hash mismatch: {name}')
    questions = [json.loads(line) for line in (source/'prompts.jsonl').read_text().splitlines()]
    records = [json.loads(line) for line in (source/'records.jsonl').read_text().splitlines()]
    expected = requested_keys(questions, SimpleNamespace(**manifest['arguments']))
    keys = [(r['id'], r['sample_index']) for r in records]
    if (len(keys) != len(set(keys)) or not set(keys).issubset(expected) or
            (manifest['status'] == 'completed' and set(keys) != expected)):
        raise ValueError('Missing/duplicate stored trajectories')
    by_id = {q['id']: q for q in questions}
    if len(by_id) != len(questions) or manifest['expected_answers'] != len(expected):
        raise ValueError('Prompt identities or declared coverage invalid')
    if manifest['completed_answers'] != len(records):
        raise ValueError('Declared completed count differs from saved answers')
    for record in records:
        row = by_id[record['id']]
        if record['dataset'] != row['dataset']:
            raise ValueError('Stored answer dataset mismatch')
        if record['seed'] != manifest['arguments']['seed']+row['seed_index']*1000+record['sample_index']:
            raise ValueError('Stored generation seed mismatch')
        if len(record['completion_tokens']) != record['output_tokens']:
            raise ValueError('Stored completion token count mismatch')
        if not 0 < record['output_tokens'] <= manifest['caps'][row['dataset']]:
            raise ValueError('Stored completion exceeds cap')
        if record.get('finish_reason') not in ('stop', 'length'):
            raise ValueError('Stored answer has no valid completion reason')
        if record['truncated'] != (record['finish_reason'] != 'stop'):
            raise ValueError('Completion/truncation metadata disagree')
    return manifest, questions, records


def grade_saved(source, out, questions_root='questions', workers=4, sandbox=None, allow_partial=False):
    """Never load a language model or regenerate; preserve raw source artifacts."""
    from evaluate import grading_context, grade_all, save_records
    from utils.train.rewards import is_task_correct
    source, out = Path(source), Path(out)
    if out.exists():
        raise FileExistsError(out)
    manifest, questions, records = read_generation(source, allow_partial=allow_partial)
    arguments = SimpleNamespace(**manifest['arguments'])
    arguments.generation_stats = manifest['generation']
    arguments.workers, arguments.sandbox = workers, sandbox
    for record in records:
        for field in ('correct', 'parse_error', 'grading_error'):
            record.pop(field, None)
    out.mkdir(parents=True, exist_ok=False)
    if manifest['kind'] == 'benchmark':
        suite = _suite_for_grading(questions_root, manifest)
        if (manifest['status'] == 'partial' or getattr(arguments, 'resume_keys', []) or
                manifest['suite'].get('collection_shard')):
            # Grade only complete answers in this shard; do not fabricate absent
            # answers or publish a full-benchmark score for a partial collection.
            active = {r['id'] for r in records}
            grading = grading_context(suite, arguments)
            grade_collection_rows([q for q in questions if q['id'] in active], records, grading)
            results = dict(partial_collection=True, answers=len(records),
                correct=sum(r.get('correct') is True for r in records))
            (out/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        else:
            _grade_benchmark(out, questions_root, manifest, arguments, questions, records)
    else:
        by_id = {q['id']: q for q in questions}
        for record in records:
            row = by_id[record['id']]
            record['correct'] = bool(not record['truncated'] and not record.get('boundary_error') and
                is_task_correct(record['text'], row['answer'], row['task']))
        results = {}
        for source_name in sorted({q['dataset'] for q in questions}):
            subset = [r for r in records if r['dataset'] == source_name]
            results[source_name] = dict(answers=len(subset), correct=sum(r['correct'] for r in subset),
                truncated=sum(r['truncated'] for r in subset))
        (out/'results.json').write_text(json.dumps(dict(results=results, kind=manifest['kind']), indent=2)+'\n')
    save_records(out, records)
    (out/'grading-provenance.json').write_text(json.dumps(dict(source=str(source.resolve()),
         generation_manifest_sha256=digest(source/'generation-manifest.json'),
         raw_records_sha256=manifest['records_sha256'], regenerated_answers=0), indent=2)+'\n')
    if any(r.get('grading_error') or type(r.get('correct')) is not bool for r in records):
        raise RuntimeError('Saved-answer grading has infrastructure errors')
    return records


def _suite_for_grading(questions_root, manifest):
    from evaluate import load_suite
    fresh = load_suite(Path(questions_root), bool(manifest['suite'].get('smoke')))
    if (fresh['cohort_sha256'] != manifest['suite']['cohort_sha256'] or
            fresh['revisions'] != manifest['suite']['revisions']):
        raise ValueError('Benchmark/grader provenance mismatch')
    suite = dict(manifest['suite'])
    for key in ('evalplus', 'bfcl', 'ifeval', 'mbpp'):
        suite[key] = fresh[key]
    return suite


def grade_collection_rows(questions, records, grading):
    """Bound sandbox worker lifetime without splitting one prompt's samples.

    Offline shards can contain thousands of MBPP programs; the sandbox's
    existing per-process CPU/memory limits stay unchanged. The normal 400/full
    evaluation path keeps its original batching behavior.
    """
    from evaluate import grade_all
    by_id = {}
    for record in records:
        by_id.setdefault(record['id'], []).append(record)
    for start in range(0, len(questions), 32):
        subset = questions[start:start+32]
        rows = [r for q in subset for r in by_id[q['id']]]
        grade_all(subset, rows, grading)


def _grade_benchmark(out, questions_root, manifest, arguments, questions, records):
    from evaluate import grading_context, grade_all, collect, save_results, report
    suite = _suite_for_grading(questions_root, manifest)
    grading = grading_context(suite, arguments)
    grade_all(questions, records, grading)
    results, failed = collect(questions, records, arguments.samples)
    save_results(out, arguments, suite, grading, results, failed, manifest['caps'])
    print(report(results, suite, failed), flush=True)
