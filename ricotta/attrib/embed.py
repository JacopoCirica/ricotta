"""Embedding-level attribution — for models with projected modalities.

When you inject a new modality (vision/audio/DNA encoder → projection → soft
tokens placed in the LM's embedding space), those soft tokens have no token IDs,
so token-based attribution can't touch them. Gradient attribution is
embedding-native: it differentiates the target log-prob w.r.t. ``inputs_embeds``
directly, so each projected modality vector gets a score right alongside the
text tokens.

Compose ``inputs_embeds`` yourself (run your projector, concat with text
embeddings), tell attriscope which positions are modality vs text, and use
``modality_contribution`` to get a clean "modality vs text" split. Integrated
gradients with a zero / mean-embedding baseline is the principled choice here:
by completeness the scores sum to log p(answer | full input) − log p(answer |
null input), a real decomposition of how much the modality moved the output.
"""

from __future__ import annotations

import torch

from .attribute import Attribution, _empty_scores, _source_slice, _target_logprob_from_embeds
from .models import LM


def _labels(seq: int, labels: list[str] | None) -> list[str]:
    return labels if labels is not None else [f"emb_{i}" for i in range(seq)]


@torch.no_grad()
def _resolve_embed_target(lm: LM, inputs_embeds, target_pos, target_token_id):
    seq = inputs_embeds.shape[1]
    if target_pos is None:
        target_pos = seq - 1
    if not 1 <= target_pos < seq:
        raise ValueError(f"target_pos must be in [1, {seq-1}], got {target_pos}")
    if target_token_id is None:  # default: the model's own argmax prediction
        logits = lm.model(inputs_embeds=inputs_embeds).logits[0, target_pos - 1]
        target_token_id = int(logits.argmax())
    return target_pos, target_token_id


def _default_baseline(lm, inputs_embeds, baseline_embeds, kind: str):
    if baseline_embeds is not None:
        return baseline_embeds
    if kind == "pad":
        # pad/eos-token embedding broadcast over positions — a real "null input"
        # the model handles well (unlike the zero vector, whose RMSNorm Jacobian
        # is singular, producing NaN gradients on Qwen/Llama-style models).
        ids = torch.full(inputs_embeds.shape[:2], lm.baseline_token_id,
                         device=inputs_embeds.device)
        return lm.model.get_input_embeddings()(ids).detach()
    if kind == "zero":
        return torch.zeros_like(inputs_embeds)
    if kind == "mean":
        return inputs_embeds.mean(dim=1, keepdim=True).expand_as(inputs_embeds).contiguous()
    raise ValueError("baseline kind must be 'pad', 'zero', or 'mean'")


def gradient_attribution_embeds(
    lm: LM, inputs_embeds: torch.Tensor, target_pos=None, target_token_id=None,
    method: str = "input_x_gradient", source_window=None, labels=None,
) -> Attribution:
    seq = inputs_embeds.shape[1]
    target_pos, target_token_id = _resolve_embed_target(lm, inputs_embeds, target_pos, target_token_id)
    embeds = inputs_embeds.clone().detach().requires_grad_(True)
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
    lab = _labels(seq, labels)
    return Attribution(lab, target_pos, lab[target_pos], scores, method, _name(lm))


def integrated_gradients_embeds(
    lm: LM, inputs_embeds: torch.Tensor, target_pos=None, target_token_id=None,
    steps: int = 32, baseline_embeds=None, baseline: str = "pad",
    source_window=None, labels=None,
) -> Attribution:
    seq = inputs_embeds.shape[1]
    target_pos, target_token_id = _resolve_embed_target(lm, inputs_embeds, target_pos, target_token_id)
    real = inputs_embeds.detach()
    base = _default_baseline(lm, real, baseline_embeds, baseline).detach()
    total_grad = torch.zeros_like(real)
    # midpoint Riemann rule: never samples the endpoints, so a degenerate
    # baseline (e.g. zeros, with its singular RMSNorm gradient) can't poison the
    # integral, and the approximation is more accurate per step.
    for k in range(steps):
        a = (k + 0.5) / steps
        interp = (base + a * (real - base)).requires_grad_(True)
        logp = _target_logprob_from_embeds(lm, interp, target_pos, target_token_id)
        (grad,) = torch.autograd.grad(logp, interp)
        total_grad += grad.detach()
    ig = ((real - base) * (total_grad / steps))[0].sum(dim=-1)
    scores = _empty_scores(lm, seq)
    for s in _source_slice(target_pos, source_window):
        scores[s] = ig[s]
    lab = _labels(seq, labels)
    return Attribution(lab, target_pos, lab[target_pos], scores, "integrated_gradients", _name(lm))


@torch.no_grad()
def occlusion_embeds(
    lm: LM, inputs_embeds: torch.Tensor, target_pos=None, target_token_id=None,
    baseline_embeds=None, baseline: str = "pad", source_window=None,
    labels=None, batch_size: int = 16,
) -> Attribution:
    """Vector-ablation occlusion: replace each source position's embedding with
    its baseline vector (pad/zero/mean) and measure the target log-prob drop. The
    embedding-space analogue of token occlusion, valid for soft tokens."""
    seq = inputs_embeds.shape[1]
    target_pos, target_token_id = _resolve_embed_target(lm, inputs_embeds, target_pos, target_token_id)
    base = _default_baseline(lm, inputs_embeds, baseline_embeds, baseline)

    def target_logp(batch_embeds):
        logits = lm.model(inputs_embeds=batch_embeds).logits[:, target_pos - 1].float()
        return torch.log_softmax(logits, dim=-1)[:, target_token_id]

    orig = target_logp(inputs_embeds)[0]
    sources = list(_source_slice(target_pos, source_window))
    scores = _empty_scores(lm, seq)
    for i in range(0, len(sources), batch_size):
        chunk = sources[i:i + batch_size]
        batch = inputs_embeds.repeat(len(chunk), 1, 1).clone()
        for r, s in enumerate(chunk):
            batch[r, s] = base[0, s]
        occ = target_logp(batch)
        for r, s in enumerate(chunk):
            scores[s] = orig - occ[r]
    lab = _labels(seq, labels)
    return Attribution(lab, target_pos, lab[target_pos], scores, "occlusion", _name(lm))


_EMBED_METHODS = {
    "input_x_gradient": lambda lm, e, **kw: gradient_attribution_embeds(lm, e, method="input_x_gradient", **kw),
    "saliency": lambda lm, e, **kw: gradient_attribution_embeds(lm, e, method="saliency", **kw),
    "integrated_gradients": integrated_gradients_embeds,
    "occlusion": occlusion_embeds,
}


def attribute_embeds(lm: LM, inputs_embeds: torch.Tensor, method: str = "integrated_gradients",
                     **kw) -> Attribution:
    """Attribute a target token to positions of a precomposed inputs_embeds.
    Default method is integrated_gradients (completeness → comparable scores)."""
    if method not in _EMBED_METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {list(_EMBED_METHODS)}")
    return _EMBED_METHODS[method](lm, inputs_embeds, **kw)


def text_to_embeds(lm: LM, input_ids: torch.Tensor) -> torch.Tensor:
    """Embed text token ids — handy for concatenating with projected modality
    vectors when composing inputs_embeds."""
    return lm.model.get_input_embeddings()(input_ids).detach()


def modality_contribution(attr: Attribution, modality_mask: torch.Tensor,
                          signed: bool = False) -> dict:
    """Split total attribution between modality and text positions.

    modality_mask: bool tensor (seq,), True at projected-modality positions.
    Returns absolute and fractional contribution for each side; fractions use
    |score| so opposing-sign tokens don't cancel a side to zero.
    """
    s = attr.scores.clone()
    valid = ~torch.isnan(s)
    mask = modality_mask.to(s.device) & valid
    text = (~modality_mask.to(s.device)) & valid
    mag = s.abs()
    m_abs = float(mag[mask].sum())
    t_abs = float(mag[text].sum())
    total = m_abs + t_abs or 1.0
    out = {
        "modality_abs": m_abs,
        "text_abs": t_abs,
        "modality_frac": m_abs / total,
        "text_frac": t_abs / total,
        "n_modality": int(mask.sum()),
        "n_text": int(text.sum()),
    }
    if signed:
        out["modality_net"] = float(s[mask].sum())
        out["text_net"] = float(s[text].sum())
    return out


def _name(lm: LM) -> str:
    return getattr(lm.model.config, "_name_or_path", lm.model.__class__.__name__)
