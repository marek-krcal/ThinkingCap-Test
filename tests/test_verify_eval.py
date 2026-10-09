"""Synthetic artifact checks; these tests do not pretend to run inference."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from cloud.verify_eval import verify


class CompletionAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.questions = [{'id': str(i), 'dataset': 'svamp', 'seed_index': i} for i in range(4)]
        raw = ''.join(json.dumps(q) + '\n' for q in self.questions).encode()
        (self.root / 'questions.jsonl').write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        (self.root / 'manifest.json').write_text(json.dumps({
            'cohort_sha256': digest, 'counts': {'svamp': 4}, 'revisions': {}}))
        self.records = [{'id': q['id'], 'dataset': 'svamp', 'sample_index': 0,
            'seed': 9001 + q['seed_index']*1000, 'correct': True, 'output_tokens': 1,
            'text': 'synthetic answer', 'truncated': False} for q in self.questions]
        self.results = {'cohort_sha256': digest, 'cohort_total': 4, 'samples': 1,
            'selection': 'full', 'smoke': False, 'model': 'Qwen/Qwen3-0.6B',
            'revision': 'c1899de289a04d12100db370d81485cdf75e47ca', 'adapter': None,
            'sampling': {'temperature': .6, 'top_p': .95, 'top_k': 20}, 'seed': 9001,
            'revisions': {}, 'max_tokens': {'svamp': 4096}, 'failed_benchmarks': [],
            'results': {'svamp': {'n': 4, 'prompts': 4, 'ungraded': 0, 'accuracy': 1}},
            'generation': {'responses': 4, 'output_tokens': 4}}

    def check(self):
        (self.root / 'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.records))
        (self.root / 'results.json').write_text(json.dumps(self.results))
        return verify(self.root, self.root, expected_questions=4)

    def test_complete_artifacts_pass(self):
        self.assertTrue(self.check()['verified'])

    def test_duplicate_and_missing_answers_fail(self):
        self.records[-1] = copy.deepcopy(self.records[0])
        self.assertIn('Missing or duplicate answers', self.check()['errors'])

    def test_changed_sample_count_or_seed_fails(self):
        self.results['samples'] = 4
        self.records[0]['seed'] += 1
        errors = self.check()['errors']
        self.assertIn('Wrong number of samples', errors)
        self.assertIn('Answer seed mismatch', errors)

    def test_grader_errors_fail(self):
        self.records[0]['grading_error'] = 'test failure'
        self.assertIn('Missing grade or grader error', self.check()['errors'])

    def test_score_or_token_cap_change_fails(self):
        self.results['results']['svamp']['accuracy'] = .5
        self.results['max_tokens']['svamp'] = 100
        errors = self.check()['errors']
        self.assertIn('Score mismatch: svamp', errors)
        self.assertIn('Token caps changed', errors)

    def test_missing_text_fails(self):
        self.records[0].pop('text')
        self.assertIn('Missing answer text', self.check()['errors'])
