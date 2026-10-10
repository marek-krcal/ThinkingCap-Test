import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cloud.optimize_eval import choose, command
from evaluate import generate_continuous


class OptimizationTests(unittest.TestCase):
    def test_choose_rejects_failed_and_drifting_profiles(self):
        rows = [dict(name='ok', verified=True, tokens_per_second=2),
                dict(name='bad', verified=False, tokens_per_second=100),
                dict(name='drift', verified=True, tokens_per_second=200, quality_warnings=['drift'])]
        self.assertEqual(choose(rows)['name'], 'ok')
        with self.assertRaises(ValueError):
            choose(rows[1:])

    def test_commands_keep_frozen_recipe_and_expose_only_requested_knobs(self):
        args = command(dict(dp=2, seqs=64, continuous=True), Path('/tmp/test'))
        for flag, value in [('--dtype','half'),('--samples','1'),('--limit-per-benchmark','100'),
                            ('--tensor-parallel-size','1'),('--data-parallel-size','2'),('--max-num-seqs','64')]:
            self.assertEqual(args[args.index(flag)+1], value)
        self.assertIn('--continuous-queue', args)

    def test_continuous_queue_saves_completion_order_but_returns_job_order(self):
        class Engine:
            def __init__(self):
                self.jobs = []
            def add_request(self, request_id, prompt, settings, **kw):
                self.jobs.append(request_id)
            def has_unfinished_requests(self):
                return bool(self.jobs)
            def step(self):
                request_id = self.jobs.pop()
                return [SimpleNamespace(request_id=request_id, finished=True)]
        jobs = [{'row': {'id':str(i), 'dataset':'svamp'},'seed':9001+i,'sample':0,'ids':[i]}
                for i in range(4)]
        modules = {'vllm': SimpleNamespace(SamplingParams=lambda **kw: kw),
                   'vllm.sampling_params': SimpleNamespace(RequestOutputKind=SimpleNamespace(FINAL_ONLY='final'))}
        def record(job, *args):
            return {'id':job['row']['id'],'seed':job['seed']}
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', modules), patch('evaluate.to_record', record):
            args = SimpleNamespace(max_tokens=None)
            out = Path(directory)
            records = generate_continuous(Engine(), jobs, args, out, None, {0}, 1, 2, None)
            self.assertEqual([r['id'] for r in records], ['0','1','2','3'])
            stored = [json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
            self.assertEqual([r['id'] for r in stored], ['3','2','1','0'])
