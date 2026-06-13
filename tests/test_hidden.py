"""Tests for hidden-state analysis: logit lens, hidden_states accessor, CKA."""

import numpy as np
import pytest

from ricotta import linear_cka

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


# ---- CKA (no model needed) -------------------------------------------------
def test_cka_identity_and_invariances():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8))
    assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-6)
    # CKA is invariant to orthogonal transforms and isotropic scaling
    Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert linear_cka(X, X @ Q) == pytest.approx(1.0, abs=1e-5)
    assert linear_cka(X, 3.0 * X) == pytest.approx(1.0, abs=1e-5)


def test_cka_unrelated_is_low():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 16))
    Y = rng.normal(size=(200, 16))
    assert linear_cka(X, Y) < 0.3       # independent gaussians -> low alignment


def test_cka_drift_monotone_with_noise():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 16))
    near = X + 0.1 * rng.normal(size=X.shape)
    far = X + 2.0 * rng.normal(size=X.shape)
    assert linear_cka(X, near) > linear_cka(X, far)   # more perturbation -> more drift


# ---- model-backed (tiny Qwen3) ---------------------------------------------
@pytest.fixture(scope="module")
def lm():
    from ricotta import LM
    try:
        return LM.load(TINY, device="cpu", eager_attention=True)
    except OSError:
        pytest.skip("tiny model unreachable (offline?)")


def test_hidden_states_accessor_shape(lm):
    ids = lm.encode("hello world example")
    hs = lm.hidden_states(ids)
    n_layers = lm.model.config.num_hidden_layers
    assert hs.shape[0] == n_layers + 1          # embeddings + each block
    assert hs.shape[1] == ids.shape[1]
    assert str(hs.device) == "cpu"


def test_logit_lens_runs(lm):
    from ricotta import logit_lens
    ll = logit_lens(lm, "the quick brown fox", top_k=3)
    assert len(ll.layers) == lm.model.config.num_hidden_layers
    traj = ll.trajectory(pos=-1)
    assert len(traj) == len(ll.layers) and all(isinstance(t, str) for t in traj)


def test_cka_drift_self_is_one(lm):
    from ricotta import cka_drift
    d = cka_drift(lm, lm, ["hello there", "a b c d e"])
    assert np.nanmin(d.cka) > 0.99             # same model vs itself -> CKA ~1
