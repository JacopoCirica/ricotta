"""Tests specific to the combined ricotta package (umbrella API + integration)."""

import torch

import ricotta
from ricotta.attrib.diff import TokenDiff


def test_umbrella_exposes_all_three_layers():
    # attn + attrib always; circuits only with the extra
    for name in ["get_attention", "head_view",                       # attn
                 "LM", "token_logprob_diff", "cot_faithfulness",      # attrib
                 "attribute_embeds", "posttrain_report"]:
        assert hasattr(ricotta, name), name
    assert isinstance(ricotta.HAS_CIRCUITS, bool)
    if ricotta.HAS_CIRCUITS:
        for name in ["diff_graphs", "to_markdown", "attribute_checkpoint"]:
            assert hasattr(ricotta, name), name


def test_tokendiff_top_excludes_nan_positions():
    # position 0 has NaN delta (nothing predicts the first token)
    delta = torch.tensor([float("nan"), 0.5, -0.3, 0.0])
    d = TokenDiff(["a", "b", "c", "d"], delta, "logprob_diff", "A", "B")
    top = d.top(4)
    assert all(v == v for _, _, v in top)        # no NaN values returned
    assert 0 not in [i for i, _, _ in top]       # NaN position never selected


def test_subpackages_importable():
    from ricotta import attn, attrib
    assert hasattr(attn, "get_attention") and hasattr(attrib, "LM")
