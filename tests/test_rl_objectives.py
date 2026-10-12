import unittest
import torch
from utils.train.objectives import (binary_rewards, leave_one_out, preference_pairs,
                                   simpo_loss, rloo_loss, completion_logprob_sum)


class BinaryObjectives(unittest.TestCase):
    def test_binary_grading_and_infrastructure_failure(self):
        self.assertEqual(binary_rewards([{'correct': True}, {'correct': False},
                         {'correct': True, 'truncated': True}]), [1, 0, 0])
        for record in ({'correct': None}, {'correct': False, 'grading_error': 'broken'}):
            with self.assertRaises(ValueError):
                binary_rewards([record])

    def test_rloo_is_leave_one_out_not_group_mean_or_length_normalization(self):
        rewards = torch.tensor([1., 0., 1., 0.], requires_grad=True)
        advantage = leave_one_out(rewards)
        torch.testing.assert_close(advantage, torch.tensor([2/3, -2/3, 2/3, -2/3]))
        self.assertFalse(advantage.requires_grad)
        logits = torch.tensor([-10., -20., -30., -40.], requires_grad=True)
        rloo_loss(logits, rewards).backward()
        torch.testing.assert_close(logits.grad, -advantage/4)
        for values in ([0., 0.], [1., 1.]):
            torch.testing.assert_close(leave_one_out(torch.tensor(values)), torch.zeros(2))
        with self.assertRaises(ValueError):
            leave_one_out(torch.tensor([1.]))

    def test_simpo_strict_preferences_no_tie_or_length_selection(self):
        self.assertEqual(preference_pairs([1, 0, 1, 0]), [(0, 1), (0, 3), (2, 1), (2, 3)])
        self.assertEqual(preference_pairs([0, 0]), [])
        self.assertEqual(preference_pairs([1, 1]), [])
        with self.assertRaises(ValueError):
            preference_pairs([.5, 1])
        chosen = torch.tensor(-20., requires_grad=True)
        rejected = torch.tensor(-40., requires_grad=True)
        loss = simpo_loss(chosen, rejected, 10, 20)
        torch.testing.assert_close(loss, -torch.nn.functional.logsigmoid(torch.tensor(-1.)))
        loss.backward()
        self.assertLess(chosen.grad.item(), 0)
        self.assertGreater(rejected.grad.item(), 0)

    def test_prompt_boundary_mask_and_last_token_included(self):
        logits = torch.randn(1, 6, 9, requires_grad=True)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        actual = completion_logprob_sum(logits, tokens, 3)
        expected = torch.log_softmax(logits.float(), -1)[0, [2, 3, 4], [4, 5, 6]].sum()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(float(logits.grad[:, :2].abs().sum()), 0)
        self.assertEqual(float(logits.grad[:, 5:].abs().sum()), 0)

    def test_tiny_qwen_lora_forward_backward_and_sliced_logits(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(Qwen3ForCausalLM(Qwen3Config(vocab_size=32,
                   hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                   num_attention_heads=2, num_key_value_heads=2, head_dim=16)),
                   LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                              target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]])
        full = model(tokens, use_cache=False).logits
        sliced = model(tokens, use_cache=False, logits_to_keep=4).logits
        expected = completion_logprob_sum(full, tokens, 3)
        actual = completion_logprob_sum(sliced, tokens[:, 2:], 1)
        torch.testing.assert_close(actual, expected)
        (-actual).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for p in model.parameters() if p.requires_grad))
