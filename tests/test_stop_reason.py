"""Completion metadata, not one tokenizer EOS ID, determines truncation."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluate import to_record, grade_one
from cloud.recover_stops import recover


class StopReasonTests(unittest.TestCase):
    def record(self, finish, stop=None):
        output = SimpleNamespace(lora_request=None, outputs=[SimpleNamespace(
            token_ids=[123], finish_reason=finish, stop_reason=stop)])
        with patch('evaluate.record_from_tokens', return_value={'truncated': True}):
            return to_record({}, output, None, None, {999}, 1, 2)

    def test_model_eos_not_in_tokenizer_eos_is_completed(self):
        record = self.record('stop', 123)
        self.assertFalse(record['truncated'])
        self.assertEqual(record['stop_reason'], 123)
        self.assertEqual(record['completion_verified_by'], 'vllm_finish_reason')

    def test_implicit_eos_stop_is_completed(self):
        self.assertFalse(self.record('stop')['truncated'])

    def test_length_and_abort_are_not_completed(self):
        for reason in ('length', 'abort'):
            self.assertTrue(self.record(reason)['truncated'])

    def test_unknown_finish_reason_fails_closed(self):
        with self.assertRaises(ValueError):
            self.record('unexpected')

    def test_malformed_thinking_is_not_a_valid_empty_tool_answer(self):
        verdict = grade_one({'grader': 'bfcl'}, {
            'response': '', 'boundary_error': 'unbalanced_thinking', 'truncated': False}, {})
        self.assertEqual(verdict, (False, 'unbalanced_thinking'))

    def test_legacy_recovery_is_conservative_at_cap(self):
        records = [dict(id=str(i), sample_index=0, dataset='svamp',
                        output_tokens=n, truncated=True, correct=False) for i, n in enumerate((99, 100))]
        results = dict(backend='vllm', generation={'responses': 2},
                       max_model_len=4196, max_tokens={'svamp': 100})
        self.assertEqual(recover(records, results), ['0'])
        self.assertFalse(records[0]['truncated'])
        self.assertTrue(records[1]['truncated'])
        self.assertNotIn('correct', records[0])
        with self.assertRaises(ValueError):
            recover([records[0], records[0]], results)
