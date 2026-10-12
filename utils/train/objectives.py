"""Binary correctness objectives; no token reward, reference model or KL term."""
import torch
import torch.nn.functional as F


def binary_rewards(records):
    values = []
    for record in records:
        if record.get('grading_error') or type(record.get('correct')) is not bool:
            raise ValueError('A missing/broken grader is not a zero reward')
        values.append(float(record['correct'] and not record.get('truncated', False)))
    return values


def leave_one_out(rewards):
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError('RLOO requires at least two trajectories for one prompt')
    return (rewards - (rewards.sum() - rewards) / (rewards.numel() - 1)).detach()


def preference_pairs(rewards):
    """All strict binary win/loss pairs; no length preference or tie breaking."""
    if any(value not in (0, 1) for value in rewards):
        raise ValueError('Only binary correctness rewards are supported')
    return [(i, j) for i, value in enumerate(rewards) if value == 1
            for j, other in enumerate(rewards) if other == 0]


def simpo_loss(chosen_sum, rejected_sum, chosen_length, rejected_length,
               beta=2.0, gamma=1.0):
    if chosen_length <= 0 or rejected_length <= 0:
        raise ValueError('Completions cannot be empty')
    return -F.logsigmoid(beta * (chosen_sum / chosen_length -
                                rejected_sum / rejected_length) - gamma)


def rloo_loss(logprob_sums, rewards):
    """One on-policy REINFORCE update; sequence SUM, not a length reward."""
    if logprob_sums.shape != rewards.shape:
        raise ValueError('Log probabilities and rewards must align')
    return -(logprob_sums * leave_one_out(rewards)).mean()


def completion_logprob_sum(logits, tokens, prompt_length):
    """Causal prompt/completion boundary, no prompt/padding loss."""
    if not 0 < prompt_length < tokens.shape[-1]:
        raise ValueError('Need a nonempty prompt and completion')
    predictions = logits[:, prompt_length - 1:-1, :].float()
    targets = tokens[:, prompt_length:]
    # fp32 reductions avoid FP16 overflow; cross entropy avoids a second huge
    # log_softmax tensor. All generated tokens, including EOS, are actions.
    return -F.cross_entropy(predictions.reshape(-1, predictions.shape[-1]),
                            targets.reshape(-1), reduction='sum')
