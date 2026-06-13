"""Diff two circuit-tracer attribution graphs computed on the same prompt
with the same transcoders (e.g. base model vs post-trained checkpoint).

Nodes are matched exactly: transcoder features by (layer, position, feature_id)
— the feature space is shared because both graphs use the same transcoders —
and error/embed nodes by (layer, position). Per-node *influence* (total direct
+ indirect effect on the output logits, as in circuit-tracer's pruning) is
L1-normalized within each graph so the two checkpoints are comparable even if
their logit distributions have different sharpness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from circuit_tracer.graph import Graph, compute_node_influence


@dataclass
class FeatureDelta:
    layer: int
    pos: int
    feature: int
    influence_a: float
    influence_b: float
    activation_a: float | None = None
    activation_b: float | None = None

    @property
    def delta(self) -> float:
        return self.influence_b - self.influence_a


@dataclass
class LogitDelta:
    token: str
    prob_a: float
    prob_b: float

    @property
    def delta(self) -> float:
        return self.prob_b - self.prob_a


@dataclass
class GraphDiff:
    name_a: str
    name_b: str
    prompt: str
    tokens: list[str]
    features: list[FeatureDelta]          # matched + one-sided, all in one list
    logits: list[LogitDelta]
    error_influence_a: float              # total influence routed through error nodes
    error_influence_b: float
    top_feature_jaccard: float            # overlap of top-N feature sets
    summary: dict = field(default_factory=dict)

    def gained(self, n=20):
        return sorted([f for f in self.features if f.delta > 0], key=lambda f: -f.delta)[:n]

    def lost(self, n=20):
        return sorted([f for f in self.features if f.delta < 0], key=lambda f: f.delta)[:n]

    def new_in_b(self, n=20):
        return sorted([f for f in self.features if f.influence_a == 0.0],
                      key=lambda f: -f.influence_b)[:n]

    def dropped_from_a(self, n=20):
        return sorted([f for f in self.features if f.influence_b == 0.0],
                      key=lambda f: -f.influence_a)[:n]


def _node_influence(graph: Graph) -> torch.Tensor:
    """Influence of every node on the output logits, L1-normalized."""
    n_nodes = graph.adjacency_matrix.shape[0]
    logit_weights = torch.zeros(n_nodes, device=graph.adjacency_matrix.device)
    logit_weights[-len(graph.logit_targets):] = graph.logit_probabilities
    infl = compute_node_influence(graph.adjacency_matrix.float(), logit_weights.float())
    total = infl.abs().sum()
    return infl / total if total > 0 else infl


def _feature_maps(graph: Graph, influence: torch.Tensor):
    """(layer, pos, feature) -> (influence, activation) for active feature nodes."""
    feats = graph.active_features  # (n, 3): layer, pos, feature_idx
    n_feat = feats.shape[0]
    fmap: dict[tuple[int, int, int], tuple[float, float | None]] = {}

    acts = None
    if graph.activation_values is not None and graph.selected_features is not None:
        acts = torch.zeros(n_feat)
        sel = graph.selected_features.cpu()
        acts[sel] = graph.activation_values.cpu().float()

    feats_cpu = feats.cpu()
    infl_cpu = influence[:n_feat].cpu()
    for i in range(n_feat):
        layer, pos, feat = (int(x) for x in feats_cpu[i])
        fmap[(layer, pos, feat)] = (
            float(infl_cpu[i]),
            float(acts[i]) if acts is not None else None,
        )
    return fmap


def _error_influence(graph: Graph, influence: torch.Tensor) -> float:
    """Total influence mass on error nodes — how much of the computation the
    transcoders fail to explain. If this differs a lot between checkpoints,
    the transcoders transfer poorly and the diff is unreliable."""
    n_feat = graph.active_features.shape[0]
    n_pos = graph.n_pos
    n_layers = graph.cfg.n_layers
    return float(influence[n_feat:n_feat + n_layers * n_pos].abs().sum())


def diff_graphs(
    graph_a: Graph,
    graph_b: Graph,
    name_a: str = "base",
    name_b: str = "finetuned",
    top_n_overlap: int = 50,
) -> GraphDiff:
    if graph_a.input_string != graph_b.input_string:
        raise ValueError(
            f"graphs were computed on different prompts: "
            f"{graph_a.input_string!r} vs {graph_b.input_string!r}"
        )
    if not torch.equal(graph_a.input_tokens.cpu(), graph_b.input_tokens.cpu()):
        raise ValueError("graphs have different input tokens")

    infl_a = _node_influence(graph_a)
    infl_b = _node_influence(graph_b)
    fmap_a = _feature_maps(graph_a, infl_a)
    fmap_b = _feature_maps(graph_b, infl_b)

    features = []
    for key in fmap_a.keys() | fmap_b.keys():
        ia, aa = fmap_a.get(key, (0.0, None))
        ib, ab = fmap_b.get(key, (0.0, None))
        features.append(FeatureDelta(
            layer=key[0], pos=key[1], feature=key[2],
            influence_a=ia, influence_b=ib,
            activation_a=aa, activation_b=ab,
        ))

    # top-N overlap (positions collapsed: a "feature" = (layer, feature_id))
    def top_set(fmap):
        agg: dict[tuple[int, int], float] = {}
        for (layer, _pos, feat), (infl, _act) in fmap.items():
            agg[(layer, feat)] = agg.get((layer, feat), 0.0) + abs(infl)
        return set(sorted(agg, key=lambda k: -agg[k])[:top_n_overlap])

    top_a, top_b = top_set(fmap_a), top_set(fmap_b)
    jaccard = len(top_a & top_b) / max(1, len(top_a | top_b))

    # logit deltas over the union of both graphs' target tokens
    la = {t.token_str: float(p) for t, p in zip(graph_a.logit_targets, graph_a.logit_probabilities)}
    lb = {t.token_str: float(p) for t, p in zip(graph_b.logit_targets, graph_b.logit_probabilities)}
    logits = [LogitDelta(token=t, prob_a=la.get(t, 0.0), prob_b=lb.get(t, 0.0))
              for t in la.keys() | lb.keys()]
    logits.sort(key=lambda d: -abs(d.delta))

    from transformers import AutoTokenizer  # tokens for display only

    try:
        tok = AutoTokenizer.from_pretrained(graph_a.cfg.tokenizer_name)
        tokens = tok.convert_ids_to_tokens(graph_a.input_tokens.cpu().tolist())
    except Exception:
        tokens = [str(int(t)) for t in graph_a.input_tokens.cpu()]

    return GraphDiff(
        name_a=name_a, name_b=name_b,
        prompt=graph_a.input_string,
        tokens=tokens,
        features=features,
        logits=logits,
        error_influence_a=_error_influence(graph_a, infl_a),
        error_influence_b=_error_influence(graph_b, infl_b),
        top_feature_jaccard=jaccard,
        summary={
            "n_features_a": len(fmap_a),
            "n_features_b": len(fmap_b),
            "n_matched": len(fmap_a.keys() & fmap_b.keys()),
        },
    )
