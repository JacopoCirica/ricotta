"""Tests for True Thinking Score primitives (no model needed)."""

import random

from ricotta.tts import (
    TTSResult,
    TTSStep,
    classify,
    perturb_step,
    segment_steps,
    tts_chart,
)


def test_perturb_numeric_changes_numbers_only():
    rng = random.Random(0)
    out = perturb_step("we compute (33 + 19 + 14) - 6", rng)
    assert out is not None and out != "we compute (33 + 19 + 14) - 6"
    # non-digit characters preserved, structure intact
    assert "compute" in out and "(" in out and "+" in out
    # every integer shifted by an offset in {-3..3}\{0}
    import re
    orig = [int(x) for x in re.findall(r"-?\d+", "33 19 14 6")]
    new = [int(x) for x in re.findall(r"-?\d+", out)]
    assert len(orig) == len(new)
    assert all(abs(a - b) in (1, 2, 3) for a, b in zip(orig, new))


def test_perturb_nonnumeric_returns_none():
    assert perturb_step("therefore the conclusion follows", random.Random(0)) is None


def test_segment_steps_splits_on_paragraphs_and_markers():
    text = "First, add the values.\n\nThen we subtract. Therefore the total is 60."
    steps = segment_steps(text)
    assert len(steps) >= 2
    assert any("First" in s for s in steps) and any("Then" in s or "Therefore" in s for s in steps)


def test_classify_thresholds():
    assert classify(0.9) == "true-thinking"
    assert classify(0.001) == "decorative"
    assert classify(0.3) == "intermediate"


def test_tts_chart_renders(tmp_path):
    steps = [TTSStep(0, "add the numbers", 0.8, 0.7, 0.75, {}, "true-thinking"),
             TTSStep(1, "let me re-check", 0.002, 0.001, 0.0015, {}, "decorative", self_verification=True)]
    r = TTSResult(problem="q", answer="42", steps=steps, model_name="test")
    svg = tts_chart(r, html_file=str(tmp_path / "tts.html"))
    assert svg.startswith("<svg") and "0.750" in svg
    assert len(r.true_thinking()) == 1 and len(r.decorative()) == 1
