"""Checkpoint diffing at the token level.

Two complementary signals over the same sequence, run through two checkpoints
(e.g. base vs post-trained, teacher vs student):

- ``token_logprob_diff``: per-position difference in log p(actual token). This
  is the green/red heatmap from contrastive self-distillation work — where does
  model B assign more/less probability to what was actually said?
- ``token_kl``: per-position KL between the two next-token distributions. A
  symmetric "how much did the predictive distribution move here", less noisy
  than a single-token logprob when you care about behavior, not one continuation.

``attribution_diff`` goes a level deeper: attribute the same target under both
models and subtract, giving "which input tokens does B rely on that A didn't".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .attribute import Attribution, attribute
from .models import LM


@dataclass
class TokenDiff:
    tokens: list[str]
    delta: torch.Tensor          # (seq,) per-position signal, NaN at pos 0
    kind: str                    # 'logprob_diff' | 'kl'
    name_a: str
    name_b: str

    def top(self, k: int = 15, signed: bool = True):
        valid = ~torch.isnan(self.delta)
        d = self.delta.clone()
        key = (d if not signed else d.abs())
        key[~valid] = float("-inf")          # never select NaN positions (e.g. pos 0)
        idx = key.topk(min(k, int(valid.sum()))).indices
        return [(int(i), self.tokens[i], float(self.delta[i])) for i in idx]


def _check_aligned(a: LM, b: LM, ids_a: torch.Tensor, ids_b: torch.Tensor):
    if not torch.equal(ids_a.cpu(), ids_b.cpu()):
        raise ValueError("the two models tokenized the text differently; pass a "
                         "shared input_ids tensor or use the same tokenizer")


def token_logprob_diff(lm_a: LM, lm_b: LM, input_ids: torch.Tensor) -> TokenDiff:
    """delta[i] = log p_B(x_i | x_<i) - log p_A(x_i | x_<i)."""
    lp_a = lm_a.token_logprobs(input_ids.to(lm_a.device))
    lp_b = lm_b.token_logprobs(input_ids.to(lm_b.device))
    delta = lp_b.cpu() - lp_a.cpu()
    return TokenDiff(lm_a.token_strings(input_ids), delta, "logprob_diff",
                     _name(lm_a), _name(lm_b))


def token_kl(lm_a: LM, lm_b: LM, input_ids: torch.Tensor, direction: str = "a_to_b") -> TokenDiff:
    """delta[i] = KL(p_A(·|x_<i) || p_B(·|x_<i)) at each position (nats)."""
    la = torch.log_softmax(lm_a.next_token_logits(input_ids.to(lm_a.device)), dim=-1).cpu()
    lb = torch.log_softmax(lm_b.next_token_logits(input_ids.to(lm_b.device)), dim=-1).cpu()
    if direction == "b_to_a":
        la, lb = lb, la
    kl = (la.exp() * (la - lb)).sum(dim=-1)        # (seq,), row i is dist after x_<=i
    seq = input_ids.shape[1]
    delta = torch.full((seq,), float("nan"))
    delta[1:] = kl[:-1]                            # align: row i-1 predicts token i
    return TokenDiff(lm_a.token_strings(input_ids), delta, "kl", _name(lm_a), _name(lm_b))


def attribution_diff(
    lm_a: LM, lm_b: LM, input_ids: torch.Tensor,
    method: str = "input_x_gradient", **kw,
) -> tuple[Attribution, Attribution, torch.Tensor]:
    """Attribute the same target under both models; return (attr_a, attr_b, delta)
    where delta = scores_b - scores_a over source positions."""
    attr_a = attribute(lm_a, input_ids.to(lm_a.device), method=method, **kw)
    attr_b = attribute(lm_b, input_ids.to(lm_b.device), method=method, **kw)
    delta = attr_b.scores.cpu() - attr_a.scores.cpu()
    return attr_a, attr_b, delta


def _name(lm: LM) -> str:
    return getattr(lm.model.config, "_name_or_path", lm.model.__class__.__name__)
