import math

import pytest
import torch

from ricotta.attrib import (
    LM,
    Span,
    aggregate,
    attribute_embeds,
    cot_faithfulness,
    modality_contribution,
    span_ablation,
    text_to_embeds,
)

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


# ---- embedding-level attribution ------------------------------------------
@pytest.mark.parametrize("method", ["input_x_gradient", "saliency", "integrated_gradients", "occlusion"])
def test_attribute_embeds_shapes(lm, ids, method):
    emb = text_to_embeds(lm, ids)
    attr = attribute_embeds(lm, emb, method=method, target_pos=6)
    assert attr.scores.shape[0] == ids.shape[1]
    assert attr.valid().sum() == 6


def test_attribute_embeds_matches_token_path(lm, ids):
    # embedding-level input×gradient on text embeds == token-level input×gradient
    from ricotta.attrib import attribute
    emb = text_to_embeds(lm, ids)
    a_emb = attribute_embeds(lm, emb, method="input_x_gradient", target_pos=5,
                             target_token_id=int(ids[0, 5]))
    a_tok = attribute(lm, ids, method="input_x_gradient", target_pos=5)
    v = a_emb.valid()
    assert torch.allclose(a_emb.scores[v], a_tok.scores[v], atol=1e-4)


def test_ig_completeness_converges(lm, ids):
    """The IG completeness gap |Σ scores − (logp(real)−logp(base))| should shrink
    as steps increase — validates we integrate the right path. (Exact only in the
    continuous limit; a random-init tiny model is jagged, so we test convergence,
    not a fixed tolerance.)"""
    emb = text_to_embeds(lm, ids)
    tpos, tid = 6, int(ids[0, 6])

    def logp(e):
        with torch.no_grad():
            lg = lm.model(inputs_embeds=e).logits[0, tpos - 1]
        return float(torch.log_softmax(lg.float(), -1)[tid])

    delta = logp(emb) - logp(torch.zeros_like(emb))

    def gap(steps):
        attr = attribute_embeds(lm, emb, method="integrated_gradients", target_pos=tpos,
                                target_token_id=tid, steps=steps, baseline="zero")
        return abs(float(attr.scores[attr.valid()].sum()) - delta)

    assert gap(128) < gap(8)   # finer Riemann sum → closer to completeness


def test_modality_contribution_split(lm, ids):
    emb = text_to_embeds(lm, ids)
    attr = attribute_embeds(lm, emb, method="input_x_gradient", target_pos=8)
    mask = torch.zeros(ids.shape[1], dtype=torch.bool)
    mask[:3] = True  # pretend first 3 positions are projected modality tokens
    c = modality_contribution(attr, mask)
    assert math.isclose(c["modality_frac"] + c["text_frac"], 1.0, abs_tol=1e-5)
    assert c["n_modality"] == 3


# ---- span-level / CoT ------------------------------------------------------
def test_aggregate_spans(lm, ids):
    from ricotta.attrib import attribute
    attr = attribute(lm, ids, method="saliency", target_pos=9)
    spans = [Span(0, 3, "a"), Span(3, 6, "b")]
    agg = aggregate(attr, spans)
    assert len(agg) == 2 and all(s.attribution is not None for s in agg)


def test_span_ablation_drops(lm, ids):
    spans = [Span(1, 4, "early"), Span(4, 7, "mid")]
    scores = span_ablation(lm, ids, spans, target_pos=9)
    assert len(scores) == 2
    assert all(s.ablation_drop is not None for s in scores)


def test_cot_faithfulness_summary(lm, ids):
    steps = [Span(1, 4, "step1"), Span(4, 7, "step2")]
    res = cot_faithfulness(lm, ids, answer_pos=9, step_spans=steps)
    assert 0.0 <= res["max_step_frac"] <= 1.0
    assert len(res["ranked"]) == 2
