import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cloud.run_eval import parse_arguments, evaluation_command
from evaluate import generate_continuous


class CloudPresetTests(unittest.TestCase):
    def args(self, extra):
        with patch('sys.argv', ['run_eval.py'] + extra):
            return parse_arguments()

    def test_coverage_and_answer_counts_keep_fixed_hardware(self):
        for coverage in ([], ['--full'], ['--limit-per-benchmark', '100'],
                         ['--limit-per-benchmark', '200'], ['--smoke']):
            for samples in (1, 2, 4, 8):
                cmd = evaluation_command(self.args(coverage + ['--samples', str(samples)]))
                for flag, value in [('--dtype', 'half'), ('--data-parallel-size', '2'),
                                    ('--tensor-parallel-size', '1'), ('--max-num-seqs', '64'),
                                    ('--samples', str(samples))]:
                    self.assertEqual(cmd[cmd.index(flag)+1], value)
                self.assertIn('--continuous-queue', cmd)
                self.assertIn('--cuda-graphs', cmd)
                self.assertEqual('--smoke' in cmd, '--smoke' in coverage)
                self.assertEqual('--limit-per-benchmark' in cmd, '--limit-per-benchmark' in coverage)

    def test_removed_knobs_and_invalid_selections_rejected(self):
        for extra in (['--samples', '3'], ['--limit-per-benchmark', '10'],
                      ['--full', '--limit-per-benchmark', '100'],
                      ['--smoke', '--full'], ['--dtype', 'float'],
                      ['--data-parallel-size', '1'], ['--max-num-seqs', '32'],
                      ['--batch-size', '128']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args(extra)

    def test_adapter_and_model_remain_configurable(self):
        cmd = evaluation_command(self.args(['--model', 'example/model', '--adapter', '/tmp/adapter']))
        self.assertEqual(cmd[cmd.index('--model')+1], 'example/model')
        self.assertEqual(cmd[cmd.index('--adapter')+1], '/tmp/adapter')

    def test_continuous_queue_retains_seed_and_restores_job_order(self):
        class Engine:
            def __init__(self):
                self.jobs = []
            def add_request(self, request_id, prompt, settings, **kw):
                self.jobs.append((request_id, settings['seed']))
            def has_unfinished_requests(self):
                return bool(self.jobs)
            def step(self):
                request_id, seed = self.jobs.pop()
                return [SimpleNamespace(request_id=request_id, finished=True, seed=seed)]
        jobs = [{'row': {'id': str(i), 'dataset': 'svamp'}, 'seed': 9001+i,
                 'sample': 0, 'ids': [i]} for i in range(4)]
        modules = {'vllm': SimpleNamespace(SamplingParams=lambda **kw: kw),
                   'vllm.sampling_params': SimpleNamespace(RequestOutputKind=SimpleNamespace(FINAL_ONLY='final'))}
        def record(job, output, *args):
            self.assertEqual(job['seed'], output.seed)
            return {'id': job['row']['id'], 'seed': job['seed']}
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', modules), patch('evaluate.to_record', record):
            out = Path(directory)
            records = generate_continuous(Engine(), jobs, SimpleNamespace(max_tokens=None), out, None, {0}, 1, 2, None)
            self.assertEqual([r['id'] for r in records], ['0', '1', '2', '3'])
            saved = [json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
            self.assertEqual([r['id'] for r in saved], ['3', '2', '1', '0'])
