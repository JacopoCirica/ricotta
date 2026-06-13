"""Tests for residual-stream belief probing.

The core test validates the probe grid on synthetic activations where the label
is linearly encoded ONLY at late positions / one layer — the grid must recover
that structure. No model needed; needs scikit-learn.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="needs the `probe` extra (scikit-learn)")

from ricotta.probe import BeliefResult, Question, _probe_grid


def test_probe_grid_recovers_late_encoding():
    rng = np.random.default_rng(0)
    N, L, P, d = 80, 3, 10, 16
    labels = rng.integers(0, 2, N)
    acts = rng.normal(size=(N, L, P, d)).astype(np.float32)
    # inject the label into layer 1, only at the last 3 positions
    signal = (labels[:, None] * 2 - 1).astype(np.float32)
    for pj in range(P - 3, P):
        acts[:, 1, pj, 0] += 3.0 * signal[:, 0]

    grid = _probe_grid(acts, labels, n_splits=3)
    assert grid.shape == (L, P)
    early = np.nanmean(grid[:, : P - 3])
    late_signal_layer = np.nanmean(grid[1, P - 3 :])
    assert late_signal_layer > 0.85          # label is decodable where injected
    assert early < 0.7                       # near chance where it isn't
    assert late_signal_layer > early + 0.2


def test_question_prompt_and_result():
    q = Question("Is water wet?", "Yes", "No", truth="A")
    assert "A) Yes" in q.prompt() and "Answer: A" in q.prompt()
    r = BeliefResult(accuracy=np.array([[0.5, 0.9], [0.5, 0.6]]),
                     layers=[4, 8], offsets=[-2, -1], n_questions=50,
                     baseline=0.5, label_balance=(25, 25))
    assert r.peak() == (4, -1, 0.9)
    assert list(r.temporal()) == [0.5, 0.9]
