"""Faithfulness metrics for attributions.

Attribution methods disagree and can be unfaithful — a pretty heatmap is not
evidence the highlighted tokens actually drive the prediction. These metrics
*test* that, by ablating tokens and watching the target probability.

Ablation = replacing a source token with a baseline token (pad/eos), which
keeps causal positions aligned (deleting tokens would shift every later
position and corrupt the comparison). All metrics are computed for the same
(target_pos, target_token) the attribution was produced for.

- comprehensiveness: remove the top-k% attributed tokens → how far does the
  target prob drop? Bigger drop = the highlighted tokens really mattered.
- sufficiency: keep only the top-k% attributed tokens → how much prob remains?
  More retained = those tokens are enough on their own.
- deletion / insertion AUC: ablate (or restore) tokens in attribution order and
  integrate the target-prob curve. Good attributions delete steeply (low AUC)
  and insert steeply (high AUC).
- agreement: rank correlation between two attributions over shared positions.
"""

from __future__ import annotations

import torch

from .attribute import Attribution
from .models import LM


@torch.no_grad()
def _target_prob(lm: LM, input_ids: torch.Tensor, target_pos: int, target_token_id: int,
                 ablate: torch.Tensor | None = None, baseline_id: int = 0) -> float:
    ids = input_ids.clone()
    if ablate is not None and ablate.any():
        ids[0, ablate] = baseline_id
    logits = lm.model(ids).logits[0, target_pos - 1].float()
    return float(torch.softmax(logits, dim=-1)[target_token_id])


def _ordered_sources(attr: Attribution) -> list[int]:
    """Source positions sorted by descending |score| (most important first)."""
    s = attr.scores.clone()
    valid = ~torch.isnan(s)
    idx = [i for i in range(len(s)) if valid[i]]
    return sorted(idx, key=lambda i: -abs(float(s[i])))


def comprehensiveness(lm: LM, input_ids: torch.Tensor, attr: Attribution,
                      k_frac: float = 0.2, baseline_id: int | None = None) -> float:
    baseline_id = lm.baseline_token_id if baseline_id is None else baseline_id
    tid = int(input_ids[0, attr.target_pos])
    order = _ordered_sources(attr)
    k = max(1, int(round(k_frac * len(order))))
    ablate = torch.zeros(input_ids.shape[1], dtype=torch.bool)
    ablate[order[:k]] = True
    p0 = _target_prob(lm, input_ids, attr.target_pos, tid, baseline_id=baseline_id)
    p1 = _target_prob(lm, input_ids, attr.target_pos, tid, ablate, baseline_id)
    return p0 - p1


def sufficiency(lm: LM, input_ids: torch.Tensor, attr: Attribution,
                k_frac: float = 0.2, baseline_id: int | None = None) -> float:
    baseline_id = lm.baseline_token_id if baseline_id is None else baseline_id
    tid = int(input_ids[0, attr.target_pos])
    order = _ordered_sources(attr)
    k = max(1, int(round(k_frac * len(order))))
    # ablate everything except the top-k attributed tokens
    ablate = torch.zeros(input_ids.shape[1], dtype=torch.bool)
    keep = set(order[:k])
    for i in order:
        if i not in keep:
            ablate[i] = True
    p0 = _target_prob(lm, input_ids, attr.target_pos, tid, baseline_id=baseline_id)
    p1 = _target_prob(lm, input_ids, attr.target_pos, tid, ablate, baseline_id)
    return p0 - p1


def deletion_curve(lm: LM, input_ids: torch.Tensor, attr: Attribution,
                   baseline_id: int | None = None) -> tuple[list[float], float]:
    """Target prob as we ablate sources most-important-first. Returns (curve, AUC).
    Lower AUC = more faithful (prob collapses as soon as key tokens go)."""
    baseline_id = lm.baseline_token_id if baseline_id is None else baseline_id
    tid = int(input_ids[0, attr.target_pos])
    order = _ordered_sources(attr)
    ablate = torch.zeros(input_ids.shape[1], dtype=torch.bool)
    curve = [_target_prob(lm, input_ids, attr.target_pos, tid, baseline_id=baseline_id)]
    for i in order:
        ablate[i] = True
        curve.append(_target_prob(lm, input_ids, attr.target_pos, tid, ablate, baseline_id))
    return curve, _auc(curve)


def insertion_curve(lm: LM, input_ids: torch.Tensor, attr: Attribution,
                    baseline_id: int | None = None) -> tuple[list[float], float]:
    """Start from all-baseline and restore sources most-important-first. Returns
    (curve, AUC). Higher AUC = more faithful."""
    baseline_id = lm.baseline_token_id if baseline_id is None else baseline_id
    tid = int(input_ids[0, attr.target_pos])
    order = _ordered_sources(attr)
    ablate = torch.ones(input_ids.shape[1], dtype=torch.bool)
    ablate[attr.target_pos:] = False     # never ablate the target/future (they're not sources anyway)
    curve = [_target_prob(lm, input_ids, attr.target_pos, tid, ablate, baseline_id)]
    for i in order:
        ablate[i] = False
        curve.append(_target_prob(lm, input_ids, attr.target_pos, tid, ablate, baseline_id))
    return curve, _auc(curve)


def agreement(a: Attribution, b: Attribution, kind: str = "spearman") -> float:
    """Rank correlation between two attributions over their shared valid sources."""
    valid = (~torch.isnan(a.scores)) & (~torch.isnan(b.scores))
    x = a.scores[valid].float()
    y = b.scores[valid].float()
    if x.numel() < 2:
        return float("nan")
    if kind == "spearman":
        x, y = _rank(x), _rank(y)
    return _pearson(x, y)


# ---- helpers ---------------------------------------------------------------
def _auc(curve: list[float]) -> float:
    if len(curve) < 2:
        return float(curve[0]) if curve else 0.0
    t = torch.tensor(curve)
    return float(torch.trapz(t, dx=1.0 / (len(curve) - 1)))


def _rank(v: torch.Tensor) -> torch.Tensor:
    order = v.argsort()
    ranks = torch.zeros_like(v)
    ranks[order] = torch.arange(v.numel(), dtype=v.dtype)
    return ranks


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    return float((x @ y) / denom) if denom > 0 else float("nan")
