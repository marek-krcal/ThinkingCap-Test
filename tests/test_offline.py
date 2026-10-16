import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from utils.offline import digest, grade_saved, read_generation, save_generation


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
