import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from utils.train.chunked import selected_logps, score_completion
from utils.train.rollout_worker import sampling_kwargs, unpack_completion, run_round
from train_batched_rl import build_round, group_advantages, simpo_pairs, pair_batches


class BatchedRLTests(unittest.TestCase):
    def test_chunked_projection_loss_and_gradients_match_unfused(self):
        torch.manual_seed(1)
        hidden = torch.randn(11, 7, requires_grad=True)
        weight = torch.randn(19, 7, requires_grad=True)
        targets = torch.arange(11)
        coeff = torch.linspace(-1, 1, 11)
        expected = torch.log_softmax(torch.nn.functional.linear(hidden, weight).float(), -1).gather(1, targets[:, None]).squeeze(1)
        expected_grads = torch.autograd.grad((expected*coeff).sum(), (hidden, weight))
        actual = selected_logps(hidden, weight, targets, chunk_size=3)
        actual_grads = torch.autograd.grad((actual*coeff).sum(), (hidden, weight))
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_chunked_qwen_lora_matches_causal_loss_and_gradient(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(Qwen3ForCausalLM(Qwen3Config(vocab_size=32,
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=16)),
            LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                       target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        model.eval()
        from utils.train.objectives import completion_logprob_sum
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        expected = completion_logprob_sum(model(tokens, use_cache=False).logits, tokens, 3)
        parameters = [p for p in model.parameters() if p.requires_grad]
        expected_grads = torch.autograd.grad(expected, parameters)
        actual = score_completion(model, [1, 2, 3], [4, 5, 6])
        actual_grads = torch.autograd.grad(actual, parameters)
        torch.testing.assert_close(actual, expected)
        for a, e in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a, e)

    def test_rollout_sampler_is_unfiltered_and_explicit_stop_policy(self):
        settings = sampling_kwargs(42, 1024, [151645])
        self.assertEqual((settings['temperature'], settings['top_p'], settings['top_k']), (1, 1, -1))
        self.assertTrue(settings['ignore_eos'])
        self.assertEqual(settings['stop_token_ids'], [151645])
        self.assertEqual(settings['min_p'], 0)

    def test_stopping_metadata_and_selected_logprobs(self):
        output = SimpleNamespace(token_ids=[3, 9], finish_reason='stop', stop_reason=9,
              logprobs=[{3: SimpleNamespace(logprob=-2.)}, {9: SimpleNamespace(logprob=-1.)}])
        record = unpack_completion(output, {9}, 2)
        self.assertTrue(record['ceiling_hit'])
        self.assertEqual(record['finish_reason'], 'stop')  # cap hit is not truncation
        self.assertEqual(record['sampler_logprobs'], [-2, -1])
        output.token_ids = [3]
        output.logprobs = output.logprobs[:1]
        with self.assertRaises(ValueError):
            unpack_completion(output, {9}, 2)

    def test_round_shapes_same_question_ids_unique_seeds(self):
        tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2])
        questions = [dict(id=str(i), messages=[]) for i in range(16)]
        jobs = build_round(questions, tokenizer, 0, 32, 8, 42)
        self.assertEqual(len(jobs), 256)
        self.assertEqual(len({j['seed'] for j in jobs}), 256)
        self.assertEqual(len({j['row']['id'] for j in jobs}), 16)
        self.assertEqual(len({j['group_index'] for j in jobs}), 32)
        later = build_round(questions, tokenizer, 1, 32, 8, 42)
        self.assertFalse({j['seed'] for j in jobs} & {j['seed'] for j in later})

    def test_group_baselines_and_global_batch_denominator(self):
        rows = [dict(group_index=i//8, correct=i<8 or i==8) for i in range(16)]
        advantages, kinds = group_advantages(rows, 8)
        self.assertEqual(kinds, dict(all_correct=1, informative=1))
        torch.testing.assert_close(torch.tensor(advantages[:8]), torch.zeros(8))
        self.assertAlmostEqual(advantages[8], 1.)
        self.assertAlmostEqual(advantages[9], -1/7)
        # Skip zero terms but divide by ALL trajectories, not only informative ones.
        values = torch.randn(16, requires_grad=True)
        expected = -(values*torch.tensor(advantages)).mean()
        expected.backward()
        torch.testing.assert_close(values.grad, -torch.tensor(advantages)/16)

    def test_pairs_disjoint_one_pass_no_truncated_losers(self):
        rows = [dict(group_index=i//8, reward=int(i%8==0), truncated=False) for i in range(32)]
        rows[1]['truncated'] = True
        pairs = simpo_pairs(rows, 8)
        self.assertEqual(pairs[0], (0, 2))
        self.assertEqual(len(pairs), 4)
        batches = pair_batches(pairs)
        self.assertEqual(len(batches), 4)
        self.assertEqual(sorted(p for batch in batches for p in batch), sorted(pairs))
        self.assertEqual(pair_batches([]), [])

    def test_worker_swaps_adapter_and_verifies_actual_input_route(self):
        class Engine:
            def __init__(self):
                self.pending = []
                self.output_processor = SimpleNamespace(request_states={})
                self.removed, self.added = [], []
            def has_unfinished_requests(self):
                return bool(self.pending)
            def remove_lora(self, adapter):
                self.removed.append(adapter)
                return True
            def add_lora(self, adapter):
                self.added.append(adapter.lora_int_id)
                return True
            def add_request(self, rid, prompt, params, lora_request):
                self.pending.append(rid)
                self.output_processor.request_states[rid] = SimpleNamespace(lora_name=lora_request.lora_name)
            def step(self):
                rid = self.pending.pop()
                completion = SimpleNamespace(token_ids=[9], finish_reason='stop', stop_reason=9,
                                              logprobs=[{9: SimpleNamespace(logprob=-1.)}])
                return [SimpleNamespace(request_id=rid, finished=True, outputs=[completion])]
        modules = {'vllm': SimpleNamespace(SamplingParams=lambda **kw: kw),
            'vllm.lora.request': SimpleNamespace(LoRARequest=lambda name, i, path:
                 SimpleNamespace(lora_name=name, lora_int_id=i)),
            'vllm.sampling_params': SimpleNamespace(RequestOutputKind=SimpleNamespace(FINAL_ONLY='final'))}
        engine = Engine()
        with tempfile.TemporaryDirectory() as directory, patch.dict('sys.modules', modules):
            weights = Path(directory)/'adapter_model.safetensors'
            weights.write_bytes(b'synthetic snapshot')
            request = dict(adapter=directory, adapter_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                round=1, cap=1024, eos_ids=[9], jobs=[dict(seed=1, prompt_tokens=[1])])
            reply, adapter = run_round(SimpleNamespace(llm_engine=engine), request, 1)
            self.assertEqual(engine.removed, [1])
            self.assertEqual(engine.added, [2])
            self.assertEqual(adapter, 2)
            self.assertEqual(reply['answers'][0]['adapter_route_verified_by'], 'engine_request_state')
            request['adapter_sha256'] = 'wrong'
            with self.assertRaises(RuntimeError):
                run_round(SimpleNamespace(llm_engine=engine), request, 2)
