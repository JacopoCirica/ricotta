"""Tests for hybrid effective-attention reconstruction.

The core test validates the reconstruction *algorithm* on synthetic tensors via
the model's own Gated DeltaNet recurrence — no 4B model download. It needs
transformers >= 4.58 (qwen3_5 modeling); skipped otherwise.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5",
                    reason="needs transformers>=4.58 for qwen3_5 modeling")


def test_effective_attention_reconstruction_is_exact():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    torch.manual_seed(0)
    T, H, Dk, Dv = 12, 3, 16, 16
    q = torch.randn(1, T, H, Dk)
    k = torch.randn(1, T, H, Dk)
    v = torch.randn(1, T, H, Dv)
    g = -torch.rand(1, T, H)                 # log-decay (negative)
    beta = torch.rand(1, T, H)

    # "real" output from the actual recurrence
    real, _ = mq.torch_recurrent_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=None,
        output_final_state=False, use_qk_l2norm_in_kernel=True)

    # reconstruct A via indicator probes, then check o == A @ v
    from ricotta.attn.hybrid import _reconstruct_gated_deltanet
    cap = dict(q=q, k=k, v=v, g=g, beta=beta, l2=True, core=real)
    A, rel_err = _reconstruct_gated_deltanet(cap, mq.torch_recurrent_gated_delta_rule)

    assert A.shape == (H, T, T)
    assert rel_err < 1e-4                                    # exact up to fp precision
    assert np.abs(np.triu(A, 1)).max() < 1e-5               # causal (no future leakage)
    o_recon = np.einsum("htj,jhc->thc", A, v[0].numpy())
    assert np.abs(o_recon - real[0].numpy()).max() < 1e-4   # reconstructs real output


def test_hybrid_data_structure():
    from ricotta.attn.hybrid import HybridAttentionData, LayerAttention

    d = HybridAttentionData(
        tokens=["a", "b", "c"],
        layers=[LayerAttention(0, "linear", np.zeros((2, 3, 3), np.float32), recon_error=1e-7),
                LayerAttention(1, "full", np.zeros((4, 3, 3), np.float32))],
        model_name="test")
    assert d.seq_len == 3
    assert len(d.linear()) == 1 and len(d.full()) == 1
    assert d.max_recon_error() == 1e-7
