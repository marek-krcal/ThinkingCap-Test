import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from utils.offline import digest, grade_saved, read_generation, save_generation
from utils.generation import run_queue
from evaluate import build_jobs, merge_shard_records
from cloud.collect_offline import prior_keys, main as collect_main
from utils.eval.recipe import SAMPLING, MAX_TOKENS, DEFAULT_MAX_TOKENS
from evaluate import BASE_MODEL, BASE_REVISION


class OfflineTests(unittest.TestCase):
    def fixture(self, root):
        root.mkdir()
        question = dict(id='gsm8k/1', dataset='gsm8k', seed_index=1000000,
            messages=[dict(role='user', content='2+2?')], answer='4', task='math', split='train')
        args = SimpleNamespace(model='fake/model', revision='pinned', samples=1,
            seed=9001, max_tokens=None, generation_stats={'seconds': 1})
        record = dict(id=question['id'], dataset='gsm8k', sample_index=0,
            seed=9001+1000000*1000, completion_tokens=[1, 2], output_tokens=2,
            finish_reason='stop', stop_reason=3, truncated=False, text='Final answer: 4')
        (root/'records.jsonl').write_text(json.dumps(record)+'\n')
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [8, 9])
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=tokenizer):
            save_generation(root, args, {}, [question], [record], kind='training')
        return question, record

    def test_round_trip_keeps_tokens_split_and_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'raw'
            self.fixture(root)
            manifest, questions, records = read_generation(root)
            self.assertEqual(manifest['completed_answers'], 1)
            self.assertEqual(questions[0]['prompt_tokens'], [8, 9])
            self.assertEqual(questions[0]['split'], 'train')
            self.assertEqual(records[0]['completion_tokens'], [1, 2])

    def test_grading_reads_saved_data_without_regenerating_or_mutating(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'raw'
            self.fixture(root)
            before = digest(root/'records.jsonl')
            with patch('evaluate.generate', side_effect=AssertionError('Must not generate')), \
                 patch('utils.train.rewards.is_task_correct', return_value=True):
                rows = grade_saved(root, Path(tmp)/'graded')
            self.assertTrue(rows[0]['correct'])
            self.assertEqual(digest(root/'records.jsonl'), before)
            provenance = json.loads((Path(tmp)/'graded/grading-provenance.json').read_text())
            self.assertEqual(provenance['regenerated_answers'], 0)

    def test_modified_artifacts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'raw'
            self.fixture(root)
            with (root/'records.jsonl').open('a') as stream:
                stream.write('{}\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                read_generation(root)

    def test_seed_cap_duplicates_and_tokens_checked_even_with_valid_hash(self):
        for field, value in [('seed', 7), ('output_tokens', 5000),
                             ('completion_tokens', []), ('finish_reason', None)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)/'raw'
                _, record = self.fixture(root)
                record[field] = value
                (root/'records.jsonl').write_text(json.dumps(record)+'\n')
                manifest = json.loads((root/'generation-manifest.json').read_text())
                manifest['records_sha256'] = digest(root/'records.jsonl')
                (root/'generation-manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    read_generation(root)

    def test_incomplete_never_reported_as_full_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'raw'
            self.fixture(root)
            manifest = json.loads((root/'generation-manifest.json').read_text())
            manifest['status'] = 'generating'
            (root/'generation-manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                read_generation(root)

    def test_deadline_retains_only_finished_answers_and_aborts_pending(self):
        class Engine:
            def __init__(self): self.pending = []; self.aborted = []
            def add_request(self, rid, *a, **kw): self.pending.append(rid)
            def has_unfinished_requests(self): return bool(self.pending)
            def step(self):
                rid = self.pending.pop(0)
                return [SimpleNamespace(request_id=rid, finished=True)]
            def abort_request(self, rid):
                self.aborted.append(rid); self.pending.remove(rid)
        engine = Engine()
        jobs = [dict(ids=[i]) for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp, patch('utils.generation.time.time', side_effect=[0, 11]):
            records = run_queue(engine, jobs, lambda j: {}, lambda j,o,r: {'id': r},
                Path(tmp)/'records.jsonl', deadline=10)
        self.assertEqual(records, [{'id': 'eval-0'}])
        self.assertEqual(set(engine.aborted), {'eval-1', 'eval-2'})

    def test_partial_shard_is_explicit_and_strict_reads_refuse_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'raw'
            question, record = self.fixture(root)
            args = SimpleNamespace(model='fake', revision='pinned', samples=2,
                seed=9001, max_tokens=None, allow_partial=True, resume_keys=[])
            manifest = save_generation(root, args, {}, [question], [record], kind='train')
            self.assertEqual(manifest['status'], 'partial')
            with self.assertRaises(ValueError): read_generation(root)
            self.assertEqual(len(read_generation(root, allow_partial=True)[2]), 1)

    def test_resume_does_not_renumber_seeds_or_accept_other_recipe(self):
        row = dict(id='x', messages=[], seed_index=42)
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a,**kw: [3])
        args = SimpleNamespace(samples=8, seed=9001, resume_keys=[('x', 0), ('x', 3)])
        jobs = build_jobs([row], tokenizer, args)
        self.assertEqual([j['sample'] for j in jobs], [1,2,4,5,6,7])
        self.assertEqual(jobs[0]['seed'], 51002)
        records = [{'id': 'x', 'sample_index': 1}]
        self.assertEqual(merge_shard_records(jobs, records, allow_partial=True), records)
        with self.assertRaises(RuntimeError): merge_shard_records(jobs, records)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'index.json'
            path.write_text(json.dumps(dict(plan_sha256='a',job_id='test-000',completed_keys=[['x',0]])))
            self.assertEqual(prior_keys(path, 'a', {'id':'test-000'}, [row]), [('x',0)])
            with self.assertRaises(ValueError): prior_keys(path, 'b', {'id':'test-000'}, [row])

    def test_collection_partial_then_resume_grades_saved_delta_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)/'input'; data.mkdir()
            question = dict(id='gsm8k/1',dataset='gsm8k',seed_index=1000000,
                messages=[dict(role='user',content='2+2?')],answer='4',task='math')
            source = data/'train.jsonl'
            source.write_text(json.dumps(question)+'\n')
            plan = dict(samples=8, model=BASE_MODEL, revision=BASE_REVISION, sampling=SAMPLING,
                caps=MAX_TOKENS, default_cap=DEFAULT_MAX_TOKENS, benchmark_suite={},
                data=dict(files={'train.jsonl':dict(sha256=digest(source))}),
                jobs=[dict(id='train-000',file='train.jsonl',split='train',start=0,stop=1,prompts=1,expected_answers=8)])
            (data/'collection-plan.json').write_text(json.dumps(plan))
            tokenizer = SimpleNamespace(apply_chat_template=lambda *a,**kw: [8,9])
            sampled = []
            def generate(rows, args, out):
                jobs = build_jobs(rows,tokenizer,args)
                chosen = jobs[:3] if not args.resume_keys else jobs
                sampled.extend(j['sample'] for j in chosen)
                records = [dict(id=j['row']['id'],dataset='gsm8k',sample_index=j['sample'],seed=j['seed'],
                    completion_tokens=[1,2],output_tokens=2,finish_reason='stop',stop_reason=3,
                    truncated=False,boundary_error=False,text='Final answer: 4') for j in chosen]
                (out/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
                return records
            for attempt in (1,2):
                argv = ['collect_offline.py','--data',str(data),'--job','train-000',
                    '--expected-plan-sha',digest(data/'collection-plan.json'),
                    '--out',str(Path(tmp)/f'attempt{attempt}')]
                if attempt==2:
                    argv += ['--resume-index',str(Path(tmp)/'attempt1/collection-index.json')]
                with patch('sys.argv',argv), patch('cloud.collect_offline.generate',generate), \
                        patch('transformers.AutoTokenizer.from_pretrained',return_value=tokenizer):
                    collect_main()
            self.assertEqual(sampled,list(range(8)))
            first = json.loads((Path(tmp)/'attempt1/collection-index.json').read_text())
            second = json.loads((Path(tmp)/'attempt2/collection-index.json').read_text())
            self.assertEqual((first['status'],first['cumulative_answers']),('partial',3))
            self.assertEqual((second['status'],second['new_answers'],second['cumulative_answers']),('completed',5,8))
