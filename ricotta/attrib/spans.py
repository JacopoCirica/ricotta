"""Span-level attribution and ablation — for chain-of-thought monitorability.

Token-level attribution is too fine for reasoning analysis: you want to ask "is
*this reasoning step* load-bearing for the answer?", not "is this comma". A span
is a contiguous token range with a label (a CoT step, a retrieved passage, a
modality block). These helpers aggregate token attributions over spans and —
the key monitorability probe — ablate a whole span and measure how far the
answer probability drops.

A large drop when a reasoning step is removed = the answer genuinely depends on
that step (faithful, monitorable). A near-zero drop = the step is decorative,
and the visible CoT is not what's driving the answer. Caveat: ablation lets the
model silently recompute, so this is necessary-but-not-sufficient evidence;
pair it with paraphrase / perturbation checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .attribute import Attribution
from .faithful import _target_prob
from .models import LM


@dataclass
class Span:
    start: int
    end: int            # exclusive
    label: str = ""

    def positions(self) -> range:
        return range(self.start, self.end)


@dataclass
class SpanScore:
    span: Span
    attribution: float | None = None     # summed token attribution (if available)
    ablation_drop: float | None = None   # target-prob drop when span removed

    def __repr__(self):
        a = "-" if self.attribution is None else f"{self.attribution:+.4f}"
        d = "-" if self.ablation_drop is None else f"{self.ablation_drop:+.4f}"
        return f"SpanScore({self.span.label!r}: attr={a}, drop={d})"


def aggregate(attr: Attribution, spans: list[Span], reduce: str = "sum") -> list[SpanScore]:
    """Sum (or mean) token attribution within each span."""
    out = []
    for sp in spans:
        vals = [float(attr.scores[i]) for i in sp.positions()
                if 0 <= i < len(attr.scores) and not torch.isnan(attr.scores[i])]
        if not vals:
            agg = None
        else:
            agg = sum(vals) if reduce == "sum" else sum(vals) / len(vals)
        out.append(SpanScore(sp, attribution=agg))
    return out


def span_ablation(
    lm: LM, input_ids: torch.Tensor, spans: list[Span], target_pos: int,
    target_token_id: int | None = None, baseline_id: int | None = None,
) -> list[SpanScore]:
    """Ablate each span (replace all its tokens with a baseline token) and record
    the drop in the target probability. This is span-level comprehensiveness."""
    baseline_id = lm.baseline_token_id if baseline_id is None else baseline_id
    tid = target_token_id if target_token_id is not None else int(input_ids[0, target_pos])
    p0 = _target_prob(lm, input_ids, target_pos, tid, baseline_id=baseline_id)

    out = []
    for sp in spans:
        ablate = torch.zeros(input_ids.shape[1], dtype=torch.bool)
        for i in sp.positions():
            if i < target_pos:                  # only ablate causal sources
                ablate[i] = True
        p1 = _target_prob(lm, input_ids, target_pos, tid, ablate, baseline_id)
        out.append(SpanScore(sp, ablation_drop=p0 - p1))
    return out


def cot_faithfulness(
    lm: LM, input_ids: torch.Tensor, answer_pos: int, step_spans: list[Span],
    target_token_id: int | None = None,
) -> dict:
    """Monitorability summary: ablate each reasoning step and rank by how much it
    drives the answer token. Returns per-step drops and the fraction of total
    drop concentrated in the single most important step (a quick 'is the answer
    carried by one hidden step' signal)."""
    scores = span_ablation(lm, input_ids, step_spans, answer_pos, target_token_id)
    drops = [s.ablation_drop for s in scores]
    total = sum(max(0.0, d) for d in drops) or 1.0
    ranked = sorted(scores, key=lambda s: -(s.ablation_drop or 0.0))
    return {
        "answer_pos": answer_pos,
        "steps": scores,
        "ranked": ranked,
        "max_step_frac": max((max(0.0, d) for d in drops), default=0.0) / total,
        "load_bearing": [s.span.label for s in ranked if (s.ablation_drop or 0) > 1e-3],
    }


def cot_faithfulness_chart(result, html_file: str | None = None):
    """Horizontal bar chart of how much each reasoning step drives the answer
    (ablation drop), sorted; the dominant step is highlighted. Accepts the dict
    returned by ``cot_faithfulness`` or a list of ``SpanScore``. Displays inline
    and/or writes an SVG."""
    import html as _html

    steps = result["ranked"] if isinstance(result, dict) else list(result)
    steps = sorted(steps, key=lambda s: -(s.ablation_drop or 0.0))
    n = len(steps)
    vals = [float(s.ablation_drop or 0.0) for s in steps]
    vmax = max((abs(v) for v in vals), default=1.0) or 1.0
    W, lab, x0, row = 680, 150, 158, 26
    H = 56 + n * row
    bw = W - x0 - 60
    p = [f'<svg width="100%" viewBox="0 0 {W} {H}" role="img" font-family="ui-monospace,Menlo,monospace">',
         '<title>CoT faithfulness</title><desc>answer-probability drop when each reasoning step is ablated</desc>',
         '<text x="20" y="22" font-size="13" fill="#222">CoT faithfulness — drop in p(answer) when each reasoning step is ablated</text>',
         '<text x="20" y="38" font-size="10" fill="#888">longer bar = more load-bearing (the answer depends on it)</text>']
    for i, s in enumerate(steps):
        y = 50 + i * row
        v = vals[i]
        w = bw * abs(v) / vmax
        color = "#d2643c" if i == 0 and v > 0 else ("#6b9bd1" if v >= 0 else "#bbb")
        label = _html.escape(str(s.span.label)[:22])   # truncate raw, then escape
        p.append(f'<text x="{lab}" y="{y+13}" text-anchor="end" font-size="11" fill="#444">{label}</text>')
        p.append(f'<rect x="{x0}" y="{y+2}" width="{max(w,1):.1f}" height="15" fill="{color}" rx="2"/>')
        p.append(f'<text x="{x0+max(w,1)+5:.1f}" y="{y+13}" font-size="10" fill="#888">{v:+.3f}</text>')
    p.append("</svg>")
    svg = "".join(p)
    if html_file:
        with open(html_file, "w") as f:
            f.write(f"<!DOCTYPE html><meta charset='utf-8'><body style='background:#fff;margin:0'>"
                    f"<div style='width:680px;max-width:100%'>{svg}</div></body>")
    try:
        from IPython.display import HTML, display
        display(HTML(svg))
    except ImportError:
        if not html_file:
            raise RuntimeError("not in IPython; pass html_file=") from None
    return svg
