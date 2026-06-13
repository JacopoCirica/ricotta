"""Inline HTML token heatmaps.

Renders tokens as colored spans that flow like text, so it scales to long
sequences (unlike all-pairs attention views): a 5k-token trajectory is just a
long paragraph the browser wraps. Diverging red/green colormap for signed
signals (logprob diffs, input×gradient); sequential for magnitude (saliency,
occlusion). Displays inline in notebooks and can also write a standalone file.
"""

from __future__ import annotations

import html
import math

import torch

from .attribute import Attribution
from .diff import TokenDiff


def _color_diverging(v: float, scale: float) -> str:
    if scale <= 0 or v == 0 or math.isnan(v):
        return "transparent"
    a = max(-1.0, min(1.0, v / scale))
    if a > 0:
        return f"rgba(34, 160, 75, {0.12 + 0.7 * a:.3f})"     # green: B prefers / positive
    return f"rgba(210, 50, 50, {0.12 + 0.7 * -a:.3f})"        # red: A prefers / negative


def _color_sequential(v: float, scale: float) -> str:
    if scale <= 0 or math.isnan(v) or v <= 0:
        return "transparent"
    a = max(0.0, min(1.0, v / scale))
    return f"rgba(70, 90, 200, {0.10 + 0.75 * a:.3f})"


def _spans(tokens, values, color_fn, scale) -> str:
    out = []
    for tok, v in zip(tokens, values):
        disp = html.escape(tok).replace(" ", "&nbsp;") or "·"
        if math.isnan(v):
            out.append(f'<span class="av-t" style="opacity:.35">{disp}</span>')
        else:
            title = f"{v:+.4f}"
            out.append(
                f'<span class="av-t" title="{title}" '
                f'style="background:{color_fn(v, scale)}">{disp}</span>'
            )
    return "".join(out)


_CSS = """
<style>
.av-wrap{font-family:-apple-system,Segoe UI,Roboto,sans-serif;font-size:13px;
 line-height:2.0;border:1px solid #ddd;border-radius:8px;padding:12px;background:#fff;color:#111}
.av-head{font-size:12px;color:#666;margin-bottom:8px}
.av-t{padding:1px 0;border-radius:3px;white-space:pre-wrap}
.av-legend{font-size:11px;color:#888;margin-top:8px}
</style>
"""


def _wrap(title: str, body: str, legend: str) -> str:
    return (f'{_CSS}<div class="av-wrap"><div class="av-head">{html.escape(title)}</div>'
            f'<div class="av-body">{body}</div><div class="av-legend">{legend}</div></div>')


def _render(html_str: str, html_file: str | None):
    if html_file:
        with open(html_file, "w") as f:
            f.write(f"<!DOCTYPE html><meta charset='utf-8'><body>{html_str}</body>")
    try:
        from IPython.display import HTML, display
        display(HTML(html_str))
    except ImportError:
        if not html_file:
            raise RuntimeError("not in IPython; pass html_file=") from None


def show_diff(diff: TokenDiff, html_file: str | None = None, scale: float | None = None):
    """Heatmap of a TokenDiff (green = B over A / positive, red = A over B)."""
    vals = diff.delta.tolist()
    finite = [abs(v) for v in vals if not math.isnan(v)]
    scale = scale or (max(finite) if finite else 1.0)
    title = f"{diff.kind}:  {diff.name_a}  →  {diff.name_b}"
    legend = (f"green = {diff.name_b} higher · red = {diff.name_a} higher · "
              f"|scale| = {scale:.3f}")
    body = _spans(diff.tokens, vals, _color_diverging, scale)
    _render(_wrap(title, body, legend), html_file)


def show_attribution(attr: Attribution, html_file: str | None = None,
                     scale: float | None = None, signed: bool | None = None):
    """Heatmap of an Attribution. Signed methods (input×gradient) use the
    diverging map; magnitude methods (saliency) use the sequential map."""
    vals = attr.scores.tolist()
    finite = [v for v in vals if not math.isnan(v)]
    if signed is None:
        signed = any(v < 0 for v in finite)
    color_fn = _color_diverging if signed else _color_sequential
    scale = scale or (max((abs(v) for v in finite), default=1.0))
    title = f"{attr.method}  ·  target [{attr.target_pos}] {attr.target_token!r}"
    legend = f"highlighted = contribution to target log-prob · |scale| = {scale:.3f}"
    body = _spans(attr.tokens, vals, color_fn, scale)
    _render(_wrap(title, body, legend), html_file)
