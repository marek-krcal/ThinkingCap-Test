"""CPU-only checks for the two-GPU evaluation configuration."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evaluate import load_engine, parse_arguments, shard_jobs, merge_shard_records, graph_configuration


class ParallelConfiguration(unittest.TestCase):
    def test_data_parallel_keeps_jobs_seeds_and_order(self):
        jobs = [{'row': {'id': str(i // 4)}, 'sample': i % 4, 'seed': 9001+i}
                for i in range(20)]
        shards = shard_jobs(jobs, 2)
        self.assertEqual([len(s) for s in shards], [10, 10])
        records = [{'id': j['row']['id'], 'sample_index': j['sample'], 'seed': j['seed']}
                   for shard in reversed(shards) for j in shard]
        merged = merge_shard_records(jobs, records)
        self.assertEqual([r['seed'] for r in merged], [j['seed'] for j in jobs])
        with self.assertRaises(RuntimeError):
            merge_shard_records(jobs, records[:-1])
        with self.assertRaises(RuntimeError):
            merge_shard_records(jobs, records[:-1] + records[:1])

    def test_data_parallel_parser_and_invalid_combinations(self):
        with patch('sys.argv', ['evaluate.py', '--out', 'runs/test', '--data-parallel-size', '2']):
            self.assertEqual(parse_arguments().data_parallel_size, 2)
        for extra in (['--data-parallel-size', '0'],
                      ['--data-parallel-size', '2', '--tensor-parallel-size', '2'],
                      ['--data-parallel-size', '2', '--backend', 'transformers']):
            with patch('sys.argv', ['evaluate.py', '--out', 'runs/test'] + extra):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_arguments()

    def test_defaults_preserve_single_gpu_and_existing_batch(self):
        with patch('sys.argv', ['evaluate.py', '--out', 'runs/test']):
            args = parse_arguments()
        self.assertEqual((args.tensor_parallel_size, args.batch_size), (1, 256))

    def test_explicit_two_gpu_queue(self):
        with patch('sys.argv', ['evaluate.py', '--out', 'runs/test',
                               '--tensor-parallel-size', '2', '--batch-size', '400']):
            args = parse_arguments()
        self.assertEqual((args.tensor_parallel_size, args.batch_size), (2, 400))

    def test_nonpositive_settings_rejected(self):
        for flag in ('--tensor-parallel-size', '--batch-size'):
            with self.subTest(flag=flag), patch('sys.argv',
                    ['evaluate.py', '--out', 'runs/test', flag, '0']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_arguments()

    def test_single_engine_receives_tensor_parallel_size(self):
        engine = Mock()
        modules = {'vllm': SimpleNamespace(LLM=engine),
                   'vllm.lora': SimpleNamespace(),
                   'vllm.lora.request': SimpleNamespace(LoRARequest=Mock()),
                   'torch': SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))}
        args = SimpleNamespace(adapter=None, dtype='half', tensor_parallel_size=2,
                               model='test', revision='pinned', seed=9001)
        with patch.dict('sys.modules', modules):
            model, lora = load_engine(args, 8192)
        self.assertEqual(engine.call_count, 1)
        self.assertEqual(engine.call_args.kwargs['tensor_parallel_size'], 2)
        self.assertEqual(engine.call_args.kwargs['max_num_seqs'], 32)
        self.assertEqual(engine.call_args.kwargs['max_num_batched_tokens'], 2048)
        self.assertEqual(engine.call_args.kwargs['gpu_memory_utilization'], 0.70)
        self.assertTrue(engine.call_args.kwargs['enforce_eager'])
        self.assertTrue(engine.call_args.kwargs['disable_custom_all_reduce'])
        self.assertIs(model, engine.return_value)
        self.assertIsNone(lora)

    def test_missing_second_gpu_fails_before_engine_construction(self):
        engine = Mock()
        modules = {'vllm': SimpleNamespace(LLM=engine),
                   'vllm.lora': SimpleNamespace(),
                   'vllm.lora.request': SimpleNamespace(LoRARequest=Mock()),
                   'torch': SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))}
        args = SimpleNamespace(adapter=None, dtype='half', tensor_parallel_size=2)
        with patch.dict('sys.modules', modules), self.assertRaises(RuntimeError):
            load_engine(args, 8192)
        engine.assert_not_called()

    def test_decode_graphs_disable_eager_without_enabling_compilation(self):
        engine = Mock()
        modules = {'vllm': SimpleNamespace(LLM=engine),
                   'vllm.lora.request': SimpleNamespace(LoRARequest=Mock()),
                   'torch': SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))}
        args = SimpleNamespace(adapter=None, dtype='half', tensor_parallel_size=1,
                               model='test', revision='pinned', seed=9001,
                               cuda_graphs=True, max_num_seqs=64)
        with patch.dict('sys.modules', modules):
            load_engine(args, 8192)
        self.assertFalse(engine.call_args.kwargs['enforce_eager'])
        config = engine.call_args.kwargs['compilation_config']
        self.assertEqual(config['mode'], 0)
        self.assertEqual(config['cudagraph_mode'], 'FULL_DECODE_ONLY')
        self.assertEqual(config['cudagraph_capture_sizes'], [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64])
        self.assertEqual(config['max_cudagraph_capture_size'], 64)
        self.assertIsNone(graph_configuration(SimpleNamespace(cuda_graphs=False)))

    def test_invalid_scheduler_settings_rejected(self):
        for flag, value in (('--max-num-seqs', '0'),
                            ('--max-num-batched-tokens', '0'),
                            ('--gpu-memory-utilization', '1')):
            with self.subTest(flag=flag), patch('sys.argv',
                    ['evaluate.py', '--out', 'runs/test', flag, value]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_arguments()
