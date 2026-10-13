"""Completion-only LM-head scoring without a full sequence-by-vocabulary tensor."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunk_logps(hidden, weight, targets):
    logits = F.linear(hidden, weight).float()
    return -F.cross_entropy(logits, targets, reduction='none')


def selected_logps(hidden, weight, targets, chunk_size=128):
    if hidden.ndim != 2 or targets.ndim != 1 or len(hidden) != len(targets) or not len(targets):
        raise ValueError('Expected one hidden state per nonempty completion target')
    if chunk_size < 1:
        raise ValueError('Positive chunk size required')
    pieces = []
    for start in range(0, len(targets), chunk_size):
        h, y = hidden[start:start+chunk_size], targets[start:start+chunk_size]
        # Recompute each small LM-head projection during backward, instead of
        # retaining all vocabulary logits/cross-entropy intermediates at once.
        if torch.is_grad_enabled() and (h.requires_grad or weight.requires_grad):
            pieces.append(checkpoint(chunk_logps, h, weight, y, use_reentrant=False))
        else:
            pieces.append(chunk_logps(h, weight, y))
    return torch.cat(pieces)


def score_completion(model, prompt, completion, return_tokens=False):
    if not prompt or not completion:
        raise ValueError('Prompt and completion must be nonempty')
    device = next(model.parameters()).device
    tokens = torch.tensor([prompt+completion], device=device)
    base = model.get_base_model()
    # Calling the transformer, not CausalLM.forward, avoids its full LM head.
    hidden = base.model(tokens, use_cache=False, return_dict=True).last_hidden_state
    predictors = hidden[0, len(prompt)-1:-1]
    targets = tokens[0, len(prompt):]
    logps = selected_logps(predictors, base.get_output_embeddings().weight, targets)
    return logps if return_tokens else logps.sum()
