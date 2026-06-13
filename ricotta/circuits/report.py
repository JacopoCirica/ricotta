"""Human-readable reports for a GraphDiff."""

from __future__ import annotations

from .diff import GraphDiff

# Neuronpedia hosts dashboards for the qwen3 transcoder features; the source
# slug pattern below matches the mwhanna PLT sets. Override if yours differs.
NEURONPEDIA_URL = "https://www.neuronpedia.org/{model}/{layer}-transcoder-hp/{feature}"


def _tok(tokens, pos):
    t = tokens[pos] if 0 <= pos < len(tokens) else "?"
    return t.replace("Ġ", " ").replace("▁", " ")


def _feature_line(d, tokens, model_slug):
    url = NEURONPEDIA_URL.format(model=model_slug, layer=d.layer, feature=d.feature)
    return (
        f"| L{d.layer} F{d.feature} | `{_tok(tokens, d.pos)}` (p{d.pos}) "
        f"| {d.influence_a:+.4f} | {d.influence_b:+.4f} | {d.delta:+.4f} "
        f"| [np]({url}) |"
    )


def to_markdown(diff: GraphDiff, model_slug: str = "qwen3-0.6b", n: int = 15) -> str:
    header = "| feature | token | infl A | infl B | Δ | link |\n|---|---|---|---|---|---|"
    lines = [
        f"# circuitdiff: {diff.name_a} → {diff.name_b}",
        f"\nPrompt: `{diff.prompt}`\n",
        "## Output distribution shift\n",
        "| token | p(A) | p(B) | Δ |", "|---|---|---|---|",
    ]
    for d in diff.logits[:10]:
        lines.append(f"| `{d.token}` | {d.prob_a:.3f} | {d.prob_b:.3f} | {d.delta:+.3f} |")

    s = diff.summary
    lines += [
        "\n## Health checks\n",
        f"- Active features: {s['n_features_a']} ({diff.name_a}) / "
        f"{s['n_features_b']} ({diff.name_b}), {s['n_matched']} matched.",
        f"- Top-feature Jaccard overlap: **{diff.top_feature_jaccard:.2f}** "
        "(low overlap = heavily rewired circuit, or poor transcoder transfer).",
        f"- Error-node influence: {diff.error_influence_a:.3f} ({diff.name_a}) vs "
        f"{diff.error_influence_b:.3f} ({diff.name_b}). "
        "If B is much higher, the base transcoders explain the fine-tune poorly "
        "and feature-level deltas should be read skeptically.",
    ]

    sections = [
        (f"Features that gained influence in {diff.name_b}", diff.gained(n)),
        (f"Features that lost influence in {diff.name_b}", diff.lost(n)),
        (f"New features (inactive in {diff.name_a})", diff.new_in_b(n)),
        (f"Dropped features (inactive in {diff.name_b})", diff.dropped_from_a(n)),
    ]
    for title, rows in sections:
        lines += [f"\n## {title}\n", header]
        lines += [_feature_line(d, diff.tokens, model_slug) for d in rows]

    return "\n".join(lines)
