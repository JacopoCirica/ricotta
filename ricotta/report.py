"""Unified post-training report: what did this checkpoint change?

One call runs the attribution layer (and, if the `circuits` extra is installed,
the feature-circuit layer) over a probe set and emits a single markdown report,
optionally writing per-prompt logprob-diff heatmaps. Drop it into an eval loop
to track what each checkpoint does to the model, mechanistically.
"""

from __future__ import annotations

from pathlib import Path

from .attrib import LM, attribution_diff, show_diff, token_logprob_diff


def posttrain_report(
    base: str,
    checkpoint: str,
    prompts: list[str],
    *,
    base_arch: str | None = None,
    device=None,
    attribution_method: str = "occlusion",
    circuits: bool = False,
    out_dir: str | None = None,
) -> str:
    """Compare ``checkpoint`` against ``base`` over ``prompts``.

    base, checkpoint: HF repo ids or local paths. ``base`` and ``checkpoint``
        must share a tokenizer (true for a base and its fine-tunes).
    base_arch: architecture name for the circuits layer (defaults to ``base``);
        needed because the fine-tune loads under the base architecture.
    circuits: also run circuitdiff (requires the `circuits` extra + transcoders
        for ``base_arch``).
    out_dir: if given, write <i>.html logprob-diff heatmaps there.

    Returns the markdown report (also written to out_dir/report.md if set).
    """
    base_lm = LM.load(base, device=device)
    ckpt_lm = LM.load(checkpoint, device=device)

    out = [f"# post-training report\n", f"- base: `{base}`", f"- checkpoint: `{checkpoint}`",
           f"- prompts: {len(prompts)}\n"]

    for i, prompt in enumerate(prompts):
        ids = base_lm.encode(prompt)
        diff = token_logprob_diff(base_lm, ckpt_lm, ids)
        out.append(f"## Prompt {i}: `{_trunc(prompt)}`\n")
        out.append("Largest per-token probability shifts (Δ = checkpoint − base):\n")
        out.append("| token | Δ logp |\n|---|---|")
        for _pos, tok, d in diff.top(8):
            out.append(f"| `{tok}` | {d:+.3f} |")
        out.append("")

        # which inputs the checkpoint relies on differently for its own prediction
        try:
            _a, _b, dd = attribution_diff(base_lm, ckpt_lm, ids, method=attribution_method)
            toks = base_lm.token_strings(ids)
            ranked = sorted(
                ((float(dd[j]), toks[j]) for j in range(len(toks)) if dd[j] == dd[j]),
                key=lambda t: -abs(t[0]),
            )[:5]
            out.append("Input reliance shifts (attribution Δ):\n")
            out.append(", ".join(f"`{t}` {v:+.3f}" for v, t in ranked) + "\n")
        except Exception as e:  # attribution is best-effort; never sink the report
            out.append(f"_(attribution skipped: {e})_\n")

        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            show_diff(diff, html_file=str(Path(out_dir) / f"diff_{i}.html"))

    if circuits:
        out.append("\n## Feature-circuit diff\n")
        out.append(_circuit_section(base, checkpoint, prompts, base_arch or base))

    report = "\n".join(out)
    if out_dir:
        (Path(out_dir) / "report.md").write_text(report)
    return report


def _circuit_section(base, checkpoint, prompts, base_arch) -> str:
    from . import HAS_CIRCUITS

    if not HAS_CIRCUITS:
        return "_circuits extra not installed (`pip install ricotta[circuits]`)._"
    from .circuits import attribute_checkpoint, diff_graphs

    lines = []
    for i, prompt in enumerate(prompts):
        try:
            ga = attribute_checkpoint(base_arch, prompt, batch_size=128)
            gb = attribute_checkpoint(checkpoint, prompt, base_arch=base_arch, batch_size=128)
            gd = diff_graphs(ga, gb, name_a="base", name_b="checkpoint")
            gained = gd.gained(5)
            lines.append(f"**Prompt {i}** · top-feature Jaccard {gd.top_feature_jaccard:.2f} · "
                         f"error-node influence {gd.error_influence_a:.2f}→{gd.error_influence_b:.2f}")
            lines.append("gained features: " +
                         ", ".join(f"L{f.layer} F{f.feature} ({f.delta:+.3f})" for f in gained))
        except Exception as e:
            lines.append(f"**Prompt {i}** · circuit diff failed: {e}")
    return "\n\n".join(lines)


def _trunc(s: str, n: int = 80) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"
