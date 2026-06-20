"""Tests for ricotta.monitor (CoT monitorability)."""

import math

import numpy as np
import pytest

from ricotta.monitor import (
    GapResult,
    MonitorabilitySummary,
    bootstrap_ci,
    latent_reasoning_gap,
    monitorability_chart,
    monitorability_diff,
)

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


# ---- significance + aggregation (no model) ---------------------------------
def test_bootstrap_ci_brackets_mean():
    vals = list(np.random.default_rng(0).normal(0.5, 0.1, 200))
    lo, hi = bootstrap_ci(vals)
    assert lo < np.mean(vals) < hi and (hi - lo) < 0.1


def test_bootstrap_ci_empty():
    lo, hi = bootstrap_ci([])
    assert math.isnan(lo) and math.isnan(hi)


def _summary(name, gaps, thr=0.02):
    return MonitorabilitySummary(name, gaps, float(np.mean(gaps)), bootstrap_ci(gaps),
                                 float(np.mean([g <= thr for g in gaps])), thr, len(gaps))


def test_monitorability_diff_paired_and_significance():
    base = _summary("base", [0.5, 0.6, 0.55, 0.62, 0.58])
    other = _summary("ft", [0.1, 0.12, 0.08, 0.15, 0.09])      # clearly lower gap
    d = monitorability_diff(base, other)
    assert d.delta_mean_gap < 0                                # CoT became less load-bearing
    assert d.significant()                                     # CI excludes 0
    assert d.delta_decorative_frac >= 0                        # more decorative traces after


def test_monitorability_chart_svg(tmp_path):
    s1 = _summary("base", [0.5, 0.6, 0.55])
    s2 = _summary("ft", [0.1, 0.12, 0.08])
    svg = monitorability_chart(s1, s2, html_file=str(tmp_path / "m.html"))
    assert svg.startswith("<svg") and "latent-reasoning gap" in svg


# ---- model-backed smoke (supplied reasoning, no generation) ----------------
@pytest.fixture(scope="module")
def lm():
    from ricotta import LM
    try:
        return LM.load(TINY, device="cpu", eager_attention=True)
    except OSError:
        pytest.skip("tiny model unreachable (offline?)")


def test_latent_reasoning_gap_runs(lm):
    prompt = "Q: what is 2+2? Answer with a number.\n"
    g = latent_reasoning_gap(lm, prompt, reasoning="2 plus 2 equals 4.", answer="4")
    assert isinstance(g, GapResult)
    assert 0.0 <= g.p_with <= 1.0 and 0.0 <= g.p_without <= 1.0
    assert g.gap == pytest.approx(g.p_with - g.p_without)
