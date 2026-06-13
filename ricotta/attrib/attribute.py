"""Input-token attribution for decoder-only LMs.

Every method answers the same question: for a target position ``t`` and a
target token ``y`` (default: the actual token at ``t``), how much does each
source position ``s < t`` contribute to ``log p(y | x_<t)``? The result is a
1-D score per source position (positions ``>= t`` are NaN — they cannot affect
a causal prediction at ``t``).

Long sequences: gradient methods are a single forward+backward over the whole
context (cheap regardless of length; pair with ``gradient_checkpointing=True``
on the LM for memory). Occlusion cost scales with the number of source
positions, so restrict it with ``source_window`` to attribute only the ``w``
tokens preceding the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .models import LM


@dataclass
class Attribution:
    tokens: list[str]
    target_pos: int
    target_token: str
    scores: torch.Tensor      # (seq,), NaN outside the attributed source range
    method: str
    model_name: str = ""

    def valid(self) -> torch.Tensor:
        return ~torch.isnan(self.scores)

    def top(self, k: int = 10):
        mag = self.scores.abs()
        mag[torch.isnan(mag)] = 0.0          # exclude unattributed positions
        idx = mag.topk(min(k, int(self.valid().sum()))).indices
        return [(int(i), self.tokens[i], float(self.scores[i])) for i in idx]

    def __repr__(self):
        return (f"Attribution(method={self.method!r}, target=[{self.target_pos}] "
                f"{self.target_token!r}, n_source={int(self.valid().sum())})")


def _resolve_target(lm: LM, input_ids: torch.Tensor, target_pos: int | None,
                    target_token_id: int | None):
    seq = input_ids.shape[1]
    if target_pos is None:
        target_pos = seq - 1
    if not 1 <= target_pos < seq:
        raise ValueError(f"target_pos must be in [1, {seq-1}], got {target_pos}")
    if target_token_id is None:
        target_token_id = int(input_ids[0, target_pos])
    return target_pos, target_token_id


def _source_slice(target_pos: int, source_window: int | None) -> range:
    lo = 0 if source_window is None else max(0, target_pos - source_window)
    return range(lo, target_pos)


def _empty_scores(lm: LM, seq: int) -> torch.Tensor:
    return torch.full((seq,), float("nan"), device=lm.device)


def _target_logprob_from_embeds(lm: LM, embeds, target_pos, target_token_id):
    """log p(target_token | context) using logits at row target_pos-1."""
    logits = lm.model(inputs_embeds=embeds).logits[0]
    logprobs = torch.log_softmax(logits[target_pos - 1].float(), dim=-1)
    return logprobs[target_token_id]


def gradient_attribution(
    lm: LM,
    input_ids: torch.Tensor,
    target_pos: int | None = None,
    target_token_id: int | None = None,
    method: str = "input_x_gradient",
    source_window: int | None = None,
) -> Attribution:
    """Saliency or input×gradient of the target log-prob w.r.t. input embeddings.

    method='input_x_gradient' -> (grad · embedding) per token (signed);
    method='saliency'         -> ||grad|| per token (magnitude).
    """
    seq = input_ids.shape[1]
    target_pos, target_token_id = _resolve_target(lm, input_ids, target_pos, target_token_id)

    embed_layer = lm.model.get_input_embeddings()
    embeds = embed_layer(input_ids).clone().detach().requires_grad_(True)

    logp = _target_logprob_from_embeds(lm, embeds, target_pos, target_token_id)
    (grad,) = torch.autograd.grad(logp, embeds)

    if method == "saliency":
        per_tok = grad[0].norm(dim=-1)
    elif method == "input_x_gradient":
        per_tok = (grad[0] * embeds[0]).sum(dim=-1)
    else:
        raise ValueError(f"unknown gradient method {method!r}")

    scores = _empty_scores(lm, seq)
    for s in _source_slice(target_pos, source_window):
        scores[s] = per_tok[s].detach()
    return Attribution(lm.token_strings(input_ids), target_pos,
                       lm.token_strings(input_ids)[target_pos], scores,
                       method, getattr(lm.model.config, "_name_or_path", ""))


def integrated_gradients(
    lm: LM,
    input_ids: torch.Tensor,
    target_pos: int | None = None,
    target_token_id: int | None = None,
    steps: int = 32,
    baseline_token_id: int | None = None,
    source_window: int | None = None,
) -> Attribution:
    """Integrated Gradients: integrate the gradient along a straight path from a
    baseline embedding (default: the pad/eos token) to the real embedding.
    Satisfies completeness, so scores are comparable across tokens."""
    seq = input_ids.shape[1]
    target_pos, target_token_id = _resolve_target(lm, input_ids, target_pos, target_token_id)

    embed_layer = lm.model.get_input_embeddings()
    real = embed_layer(input_ids).detach()                       # (1, seq, d)
    base_id = lm.baseline_token_id if baseline_token_id is None else baseline_token_id
    baseline = embed_layer(torch.full_like(input_ids, base_id)).detach()

    total_grad = torch.zeros_like(real)
    for a in torch.linspace(0, 1, steps, device=lm.device):
        interp = (baseline + a * (real - baseline)).requires_grad_(True)
        logp = _target_logprob_from_embeds(lm, interp, target_pos, target_token_id)
        (grad,) = torch.autograd.grad(logp, interp)
        total_grad += grad.detach()

    avg_grad = total_grad / steps
    ig = ((real - baseline) * avg_grad)[0].sum(dim=-1)           # (seq,)

    scores = _empty_scores(lm, seq)
    for s in _source_slice(target_pos, source_window):
        scores[s] = ig[s]
    return Attribution(lm.token_strings(input_ids), target_pos,
                       lm.token_strings(input_ids)[target_pos], scores,
                       "integrated_gradients", getattr(lm.model.config, "_name_or_path", ""))


@torch.no_grad()
def occlusion(
    lm: LM,
    input_ids: torch.Tensor,
    target_pos: int | None = None,
    target_token_id: int | None = None,
    baseline_token_id: int | None = None,
    source_window: int | None = None,
    batch_size: int = 16,
) -> Attribution:
    """Occlusion: replace each source token with a baseline token and measure the
    drop in the target log-prob. score_s = logp(orig) - logp(occlude s); large
    positive = removing s hurt the prediction, so s mattered. Model-agnostic but
    costs one forward per source position — use source_window for long context."""
    seq = input_ids.shape[1]
    target_pos, target_token_id = _resolve_target(lm, input_ids, target_pos, target_token_id)
    base_id = lm.baseline_token_id if baseline_token_id is None else baseline_token_id

    def target_logp(batch_ids):
        logits = lm.model(batch_ids).logits[:, target_pos - 1].float()
        return torch.log_softmax(logits, dim=-1)[:, target_token_id]

    orig_logp = target_logp(input_ids)[0]
    sources = list(_source_slice(target_pos, source_window))

    scores = _empty_scores(lm, seq)
    for i in range(0, len(sources), batch_size):
        chunk = sources[i:i + batch_size]
        batch = input_ids.repeat(len(chunk), 1).clone()
        for r, s in enumerate(chunk):
            batch[r, s] = base_id
        occluded = target_logp(batch)
        for r, s in enumerate(chunk):
            scores[s] = orig_logp - occluded[r]
    return Attribution(lm.token_strings(input_ids), target_pos,
                       lm.token_strings(input_ids)[target_pos], scores,
                       "occlusion", getattr(lm.model.config, "_name_or_path", ""))


METHODS = {
    "input_x_gradient": lambda lm, ids, **kw: gradient_attribution(lm, ids, method="input_x_gradient", **kw),
    "saliency": lambda lm, ids, **kw: gradient_attribution(lm, ids, method="saliency", **kw),
    "integrated_gradients": integrated_gradients,
    "occlusion": occlusion,
}


def attribute(lm: LM, input_ids: torch.Tensor, method: str = "input_x_gradient", **kw) -> Attribution:
    """Dispatch to a named attribution method (see attribute.METHODS)."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {list(METHODS)}")
    return METHODS[method](lm, input_ids, **kw)
