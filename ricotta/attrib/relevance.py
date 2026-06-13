"""Relevance masking for token diffs (the RMSD recipe).

Raw per-token signals are noisy: many high-magnitude positions are unrelated
wording/style, not the behaviour you care about. This is the two-stage filter
from relevance-masked self-distillation:

1. deterministically keep the T positions of largest magnitude;
2. (optional) pass those candidates to an LLM judge that keeps up to S that are
   actually about the target behaviour.

The judge is a user-supplied callable, so attriscope has no API dependency —
plug in Claude, GPT, or a local model. ``build_judge_prompt`` gives a sensible
default prompt; the callable must return the list of *kept token indices*.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .diff import TokenDiff


def top_t_positions(diff: TokenDiff, t: int, by: str = "abs") -> list[int]:
    """The t positions of largest signal. by='abs' | 'pos' | 'neg'."""
    d = diff.delta.clone()
    d[torch.isnan(d)] = 0.0
    if by == "abs":
        key = d.abs()
    elif by == "pos":
        key = d
    elif by == "neg":
        key = -d
    else:
        raise ValueError("by must be 'abs', 'pos', or 'neg'")
    t = min(t, int((~torch.isnan(diff.delta)).sum()))
    return sorted(int(i) for i in key.topk(t).indices)


def build_judge_prompt(diff: TokenDiff, candidates: list[int], behavior: str,
                       context: int = 6) -> str:
    """A default judge prompt: show each candidate token in local context and ask
    which relate to ``behavior``."""
    lines = [
        f"We are distilling a model toward this target behaviour: {behavior!r}.",
        "Below are candidate token positions where two models disagree most. "
        "Return the positions that genuinely relate to the target behaviour "
        "(not unrelated wording/style), as a JSON list of integers.\n",
    ]
    toks = diff.tokens
    for i in candidates:
        lo, hi = max(0, i - context), min(len(toks), i + context + 1)
        window = "".join(
            f"[[{toks[j]}]]" if j == i else toks[j] for j in range(lo, hi)
        )
        lines.append(f"pos {i} (Δ={float(diff.delta[i]):+.3f}): …{window}…")
    return "\n".join(lines)


def relevance_mask(
    diff: TokenDiff,
    behavior: str,
    judge_fn: Callable[[str], list[int]],
    t: int = 32,
    by: str = "abs",
    context: int = 6,
) -> dict:
    """Full RMSD filter. Returns the candidate set, the judge-kept set, and a
    boolean mask over all positions."""
    candidates = top_t_positions(diff, t, by=by)
    prompt = build_judge_prompt(diff, candidates, behavior, context=context)
    kept = [i for i in judge_fn(prompt) if i in set(candidates)]
    mask = torch.zeros(len(diff.tokens), dtype=torch.bool)
    mask[kept] = True
    return {"candidates": candidates, "kept": sorted(kept), "mask": mask, "judge_prompt": prompt}
