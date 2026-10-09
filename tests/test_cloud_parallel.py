"""CPU-only checks for the two-GPU evaluation configuration."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evaluate import load_engine, parse_arguments


class ParallelConfiguration(unittest.TestCase):
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
