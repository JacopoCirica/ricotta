"""Unit tests for graph diffing with tiny hand-built Graphs (no models)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from circuit_tracer.attribution.targets import LogitTarget
from circuit_tracer.graph import Graph

sys.path.insert(0, str(Path(__file__).parent.parent))
from ricotta.circuits import diff_graphs, to_markdown  # noqa: E402

N_LAYERS = 2
N_POS = 3


def make_graph(feature_rows, edge_weight, logit_probs=(0.6, 0.3)):
    """Tiny graph: features -> logit edges with the given weight.

    feature_rows: list of (layer, pos, feature_id)
    Node order: [features..., errors (n_layers*n_pos), embeds (n_pos), logits].
    """
    n_feat = len(feature_rows)
    n_logits = len(logit_probs)
    n_nodes = n_feat + N_LAYERS * N_POS + N_POS + n_logits
    adj = torch.zeros(n_nodes, n_nodes)
    for i in range(n_feat):
        adj[-n_logits, i] = edge_weight  # each feature directly affects logit 0

    class FakeCfg(SimpleNamespace):
        def to_dict(self):
            return dict(vars(self))

    # mimics an HF config closely enough for circuit-tracer's UnifiedConfig
    cfg = FakeCfg(
        architectures=["FakeForCausalLM"],
        name_or_path="missing/none",
        num_hidden_layers=N_LAYERS,
        hidden_size=8,
        head_dim=2,
        num_attention_heads=4,
        intermediate_size=16,
        vocab_size=100,
        num_key_value_heads=2,
        torch_dtype="float32",
    )
    return Graph(
        input_string="test prompt",
        input_tokens=torch.tensor([1, 2, 3]),
        active_features=torch.tensor(feature_rows, dtype=torch.long),
        adjacency_matrix=adj,
        cfg=cfg,
        selected_features=torch.arange(n_feat),
        activation_values=torch.ones(n_feat),
        logit_targets=[LogitTarget(token_str=s, vocab_idx=i) for i, s in enumerate([" Austin", " Dallas"][:n_logits])],
        logit_probabilities=torch.tensor(logit_probs),
        scan_name="test-scan",
        vocab_size=100,
    )


def test_matched_features_get_deltas():
    g_a = make_graph([(0, 1, 7), (1, 2, 9)], edge_weight=1.0)
    g_b = make_graph([(0, 1, 7), (1, 2, 9)], edge_weight=1.0)
    diff = diff_graphs(g_a, g_b)
    assert diff.summary["n_matched"] == 2
    assert all(abs(f.delta) < 1e-6 for f in diff.features)
    assert diff.top_feature_jaccard == 1.0


def test_new_and_dropped_features():
    g_a = make_graph([(0, 1, 7), (1, 2, 9)], edge_weight=1.0)
    g_b = make_graph([(0, 1, 7), (1, 1, 42)], edge_weight=1.0)
    diff = diff_graphs(g_a, g_b)
    new = diff.new_in_b()
    dropped = diff.dropped_from_a()
    assert [(f.layer, f.feature) for f in new] == [(1, 42)]
    assert [(f.layer, f.feature) for f in dropped] == [(1, 9)]


def test_influence_shift_ranked():
    g_a = make_graph([(0, 1, 7), (1, 2, 9)], edge_weight=1.0)
    g_b = make_graph([(0, 1, 7), (1, 2, 9)], edge_weight=1.0)
    # strengthen feature 0's edge in B only
    g_b.adjacency_matrix[-2, 0] = 3.0
    diff = diff_graphs(g_a, g_b)
    gained = diff.gained()
    assert gained and (gained[0].layer, gained[0].feature) == (0, 7)


def test_logit_shift():
    g_a = make_graph([(0, 1, 7)], edge_weight=1.0, logit_probs=(0.6, 0.3))
    g_b = make_graph([(0, 1, 7)], edge_weight=1.0, logit_probs=(0.1, 0.8))
    diff = diff_graphs(g_a, g_b)
    by_tok = {d.token: d for d in diff.logits}
    assert by_tok[" Austin"].delta == pytest.approx(-0.5)
    assert by_tok[" Dallas"].delta == pytest.approx(0.5)


def test_prompt_mismatch_raises():
    g_a = make_graph([(0, 1, 7)], edge_weight=1.0)
    g_b = make_graph([(0, 1, 7)], edge_weight=1.0)
    g_b.input_string = "different"
    with pytest.raises(ValueError):
        diff_graphs(g_a, g_b)


def test_markdown_report_renders():
    g_a = make_graph([(0, 1, 7), (1, 2, 9)], edge_weight=1.0)
    g_b = make_graph([(0, 1, 7), (1, 1, 42)], edge_weight=2.0)
    md = to_markdown(diff_graphs(g_a, g_b))
    assert "circuitdiff" in md and "F42" in md and "Jaccard" in md
