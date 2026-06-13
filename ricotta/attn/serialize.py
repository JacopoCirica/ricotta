"""Compact serialization of attention weights for the browser.

Full attention for a modern LM is huge (layers x heads x seq^2 floats), so we
keep only the top-k keys per query row and quantize weights to uint8. Indices
are stored as uint32. Both are base64-encoded so the payload embeds directly
in the notebook HTML.
"""

from __future__ import annotations

import base64
import warnings

import numpy as np

from .extract import AttentionData

# Beyond this the notebook/browser gets sluggish; suggest narrowing layers.
PAYLOAD_WARN_BYTES = 48 * 1024 * 1024


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _topk_rows(mat: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row top-k of a [seq, seq] matrix -> (indices uint32, values uint8),
    both [seq, k], padded with index 0 / value 0 where a row has fewer than k
    nonzero entries (value 0 edges are simply not drawn)."""
    seq = mat.shape[0]
    k = min(k, seq)
    idx = np.argpartition(-mat, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(mat, idx, axis=1)
    # sort each row's top-k descending so the JS can early-out on small weights
    order = np.argsort(-vals, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    q = np.clip(np.round(vals * 255), 0, 255).astype(np.uint8)
    return idx.astype(np.uint32), q


def serialize(
    data: AttentionData,
    layers: list[int] | None = None,
    heads: list[int] | None = None,
    top_k: int = 32,
) -> dict:
    """Build the JSON-able payload consumed by the head-view JS."""
    layers = list(range(data.num_layers)) if layers is None else sorted(layers)
    heads = list(range(data.num_heads)) if heads is None else sorted(heads)
    for l in layers:
        if not 0 <= l < data.num_layers:
            raise ValueError(f"layer {l} out of range (model has {data.num_layers})")
    for h in heads:
        if not 0 <= h < data.num_heads:
            raise ValueError(f"head {h} out of range (model has {data.num_heads})")

    k = min(top_k, data.seq_len)
    est = len(layers) * len(heads) * data.seq_len * k * 5  # 4B index + 1B value
    if est > PAYLOAD_WARN_BYTES:
        warnings.warn(
            f"estimated payload ~{est / 1e6:.0f}MB; pass a smaller `layers=[...]` "
            f"subset or lower `top_k` for sequences this long",
            stacklevel=3,
        )

    attn_payload = []
    for l in layers:
        layer_entry = []
        for h in heads:
            idx, vals = _topk_rows(data.attentions[l, h], k)
            layer_entry.append({"i": _b64(idx), "v": _b64(vals)})
        attn_payload.append(layer_entry)

    return {
        "tokens": data.tokens,
        "seqLen": data.seq_len,
        "layers": layers,
        "heads": heads,
        "topK": k,
        "modelName": data.model_name,
        "attn": attn_payload,
    }
