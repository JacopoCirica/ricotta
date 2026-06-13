import math

import pytest
import torch

from ricotta.attrib import (
    LM,
    agreement,
    attribute,
    comprehensiveness,
    deletion_curve,
    insertion_curve,
    integrated_gradients,
    occlusion,
    show_attribution,
    show_diff,
    sufficiency,
    token_kl,
    token_logprob_diff,
)
from ricotta.attrib.diff import TokenDiff
from ricotta.attrib.relevance import relevance_mask, top_t_positions

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


@pytest.fixture(scope="module")
def lm():
    try:
        return LM.load(TINY, device="cpu", eager_attention=True)
    except OSError:
        pytest.skip("tiny model unreachable (offline?)")


@pytest.fixture(scope="module")
def ids(lm):
    return lm.encode("The capital of France is Paris and the sky is blue.")


# ---- attribution methods ---------------------------------------------------
@pytest.mark.parametrize("method", ["input_x_gradient", "saliency", "integrated_gradients", "occlusion"])
def test_attribution_shapes(lm, ids, method):
    attr = attribute(lm, ids, method=method, target_pos=5)
    assert attr.scores.shape[0] == ids.shape[1]
    # only source positions < target are attributed
    assert attr.valid().sum() == 5
    assert torch.isnan(attr.scores[5:]).all()
    top = attr.top(3)
    assert len(top) == 3
    # top() must never return unattributed (NaN) positions
    assert all(not math.isnan(v) and pos < 5 for pos, _, v in top)


def test_source_window_limits_attribution(lm, ids):
    attr = occlusion(lm, ids, target_pos=8, source_window=3)
    assert attr.valid().sum() == 3  # only the 3 tokens before the target


def test_integrated_gradients_completeness(lm, ids):
    # IG attributions should be finite and not all-zero
    attr = integrated_gradients(lm, ids, target_pos=6, steps=16)
    vals = attr.scores[attr.valid()]
    assert torch.isfinite(vals).all() and vals.abs().sum() > 0


# ---- diffing ---------------------------------------------------------------
def test_logprob_diff_zero_for_same_model(lm, ids):
    d = token_logprob_diff(lm, lm, ids)
    finite = d.delta[~torch.isnan(d.delta)]
    assert finite.abs().max() < 1e-4  # identical models => zero diff


def test_kl_zero_for_same_model(lm, ids):
    d = token_kl(lm, lm, ids)
    finite = d.delta[~torch.isnan(d.delta)]
    assert finite.abs().max() < 1e-4


# ---- faithfulness ----------------------------------------------------------
def test_faithfulness_metrics_run(lm, ids):
    attr = occlusion(lm, ids, target_pos=7)
    c = comprehensiveness(lm, ids, attr)
    s = sufficiency(lm, ids, attr)
    _, del_auc = deletion_curve(lm, ids, attr)
    _, ins_auc = insertion_curve(lm, ids, attr)
    for v in (c, s, del_auc, ins_auc):
        assert isinstance(v, float) and not math.isnan(v)


def test_agreement_self_is_one(lm, ids):
    attr = attribute(lm, ids, method="saliency", target_pos=6)
    assert agreement(attr, attr) == pytest.approx(1.0, abs=1e-5)


# ---- relevance + render (no model needed) ----------------------------------
def test_top_t_and_relevance_mask():
    delta = torch.tensor([float("nan"), 0.1, -0.9, 0.2, 0.8, -0.05])
    diff = TokenDiff(["a", "b", "c", "d", "e", "f"], delta, "logprob_diff", "A", "B")
    assert top_t_positions(diff, 2) == [2, 4]  # largest |Δ|
    res = relevance_mask(diff, "the topic", judge_fn=lambda p: [4], t=2)
    assert res["kept"] == [4] and res["mask"][4]


def test_render_writes_html(tmp_path):
    delta = torch.tensor([float("nan"), 0.5, -0.3, 0.0])
    diff = TokenDiff(["x", "y", "z", "w"], delta, "logprob_diff", "A", "B")
    out = tmp_path / "d.html"
    show_diff(diff, html_file=str(out))
    assert "av-wrap" in out.read_text()
