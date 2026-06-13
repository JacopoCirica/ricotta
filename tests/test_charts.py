"""Tests for the cka_drift_chart and cot_faithfulness_chart renderers."""

import numpy as np

from ricotta import cka_drift_chart, cot_faithfulness_chart
from ricotta.hidden import CKADrift
from ricotta.attrib.spans import Span, SpanScore


def test_cka_drift_chart_svg(tmp_path):
    d = CKADrift(layers=[1, 5, 23, 32], cka=np.array([0.999, 0.997, 0.889, 0.977]),
                 name_base="base", name_other="ft", n_vectors=100)
    out = tmp_path / "cka.html"
    svg = cka_drift_chart(d, html_file=str(out))
    assert svg.startswith("<svg") and "<rect" in svg
    assert "L23" in svg and "0.111" in svg          # peak layer drift annotated
    assert out.read_text().count("<rect") == 4       # one bar per layer


def test_cot_faithfulness_chart_svg(tmp_path):
    steps = [SpanScore(Span(0, 3, "framing"), ablation_drop=0.88),
             SpanScore(Span(3, 6, "setup"), ablation_drop=0.17),
             SpanScore(Span(6, 9, "case detail"), ablation_drop=0.01)]
    result = {"ranked": steps, "steps": steps}
    svg = cot_faithfulness_chart(result, html_file=str(tmp_path / "cot.html"))
    assert svg.startswith("<svg")
    assert "framing" in svg and "+0.880" in svg
    # accepts a bare list of SpanScore too
    assert cot_faithfulness_chart(steps).startswith("<svg")
