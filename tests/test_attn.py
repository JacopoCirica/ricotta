import base64

import numpy as np
import pytest
import torch

from ricotta.attn import AttentionData, from_attentions, head_view
from ricotta.attn.serialize import _topk_rows, serialize


def make_data(layers=2, heads=3, seq=10):
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(layers, heads, seq, seq)).astype(np.float32)
    # causal softmax rows
    mask = np.triu(np.full((seq, seq), -np.inf), k=1)
    attn = np.exp(logits + mask)
    attn = attn / attn.sum(-1, keepdims=True)
    tokens = [f"tok{i}" for i in range(seq)]
    return AttentionData(tokens=tokens, attentions=attn.astype(np.float32), model_name="test")


def test_attention_data_validation():
    with pytest.raises(ValueError):
        AttentionData(tokens=["a"], attentions=np.zeros((2, 3, 4, 4), dtype=np.float32))


def test_from_attentions_tuple_of_tensors():
    seq, heads = 6, 2
    per_layer = [torch.rand(1, heads, seq, seq) for _ in range(3)]
    data = from_attentions(per_layer, tokens=[str(i) for i in range(seq)])
    assert data.num_layers == 3 and data.num_heads == heads and data.seq_len == seq


def test_topk_rows_sorted_and_quantized():
    data = make_data()
    idx, vals = _topk_rows(data.attentions[0, 0], k=4)
    assert idx.shape == (10, 4) and vals.dtype == np.uint8
    # each row sorted descending
    assert all((np.diff(vals[r].astype(int)) <= 0).all() for r in range(10))
    # top-1 of last row matches argmax of the dense row
    assert idx[-1, 0] == data.attentions[0, 0, -1].argmax()


def test_serialize_roundtrip():
    data = make_data()
    payload = serialize(data, layers=[1], heads=[0, 2], top_k=4)
    assert payload["layers"] == [1] and payload["heads"] == [0, 2]
    assert len(payload["attn"]) == 1 and len(payload["attn"][0]) == 2
    raw = base64.b64decode(payload["attn"][0][0]["i"])
    idx = np.frombuffer(raw, dtype=np.uint32).reshape(10, 4)
    expected, _ = _topk_rows(data.attentions[1, 0], k=4)
    np.testing.assert_array_equal(idx, expected)


def test_serialize_rejects_bad_layer():
    with pytest.raises(ValueError):
        serialize(make_data(), layers=[99])


def test_head_view_writes_standalone_html(tmp_path):
    data = make_data()
    out = tmp_path / "view.html"
    head_view(data, html_file=str(out))
    html = out.read_text()
    assert "attnvizHeadView" in html and "tok3" in html


def test_tiny_model_end_to_end():
    from ricotta.attn import get_attention

    try:
        data = get_attention(
            "trl-internal-testing/tiny-Qwen3ForCausalLM",
            text="hello world, attention!",
            device="cpu",
        )
    except OSError:
        pytest.skip("tiny model not reachable (offline?)")
    assert data.seq_len == len(data.tokens) > 0
    # softmax rows sum to 1
    np.testing.assert_allclose(data.attentions.sum(-1), 1.0, atol=1e-4)
