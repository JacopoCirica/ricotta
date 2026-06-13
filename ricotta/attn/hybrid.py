"""Effective-attention reconstruction for hybrid linear-attention models.

Models like Qwen3.5 interleave a few softmax (full) attention layers with many
linear-attention layers (Qwen3.5 uses Gated DeltaNet). Linear-attention layers
have no softmax attention matrix, so ``output_attentions`` silently returns
nothing for them. But their output is still a *linear* function of the value
vectors:

    o_t = sum_{j<=t} A[t, j] v_j         (A[t,j] a scalar per head)

so an exact effective-attention matrix ``A`` exists. We recover it without any
closed-form derivation or approximation: capture the real (q, k, g, beta) a
layer feeds to its recurrence, then re-run that very recurrence with indicator
values (v = delta_j) — by linearity the output reads off column j of A. We
verify it by reconstruction: ``sum_j A[t,j] v_j`` reproduces the real layer
output to machine precision.

Unlike softmax attention, this A is **signed** and its rows do not sum to 1 —
the delta rule subtracts as well as adds. Treat the magnitude as "how much this
position's value flows into the output", and the sign as its direction.

Currently supports Qwen3.5 (``model_type == "qwen3_5"``). Full-attention layers
keep their real softmax weights; linear layers are reconstructed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class LayerAttention:
    layer: int                    # original index in the model
    kind: str                     # 'full' (softmax) | 'linear' (reconstructed)
    attn: np.ndarray              # (heads, seq, seq), float32
    recon_error: float | None = None   # relative reconstruction error (linear only)

    @property
    def num_heads(self) -> int:
        return self.attn.shape[0]


@dataclass
class HybridAttentionData:
    tokens: list[str]
    layers: list[LayerAttention]
    model_name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def seq_len(self) -> int:
        return len(self.tokens)

    def full(self):
        return [l for l in self.layers if l.kind == "full"]

    def linear(self):
        return [l for l in self.layers if l.kind == "linear"]

    def max_recon_error(self) -> float:
        errs = [l.recon_error for l in self.layers if l.recon_error is not None]
        return max(errs) if errs else 0.0

    def __repr__(self):
        nf, nl = len(self.full()), len(self.linear())
        return (f"HybridAttentionData(model={self.model_name!r}, layers={len(self.layers)} "
                f"[{nf} full + {nl} linear], seq_len={self.seq_len}, "
                f"max_recon_err={self.max_recon_error():.1e})")


# ---- Qwen3.5 Gated DeltaNet -------------------------------------------------
def _reconstruct_gated_deltanet(cap: dict, recurrent_fn) -> tuple[np.ndarray, float]:
    """Exact A via indicator probes, reusing the model's own recurrence.
    cap holds the captured q,k,v,g,beta and the real layer output `core`."""
    q, k, v, g, beta = cap["q"], cap["k"], cap["v"], cap["g"], cap["beta"]
    B, T, H, Dv = v.shape
    qr = q.expand(T, T, H, q.shape[-1]).contiguous()
    kr = k.expand(T, T, H, k.shape[-1]).contiguous()
    gr = g.expand(T, T, H).contiguous()
    br = beta.expand(T, T, H).contiguous()
    probe = torch.zeros(T, T, H, Dv, dtype=v.dtype, device=v.device)
    for j in range(T):
        probe[j, j, :, 0] = 1.0
    with torch.no_grad():
        core_probe, _ = recurrent_fn(
            qr, kr, probe, g=gr, beta=br, initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=cap["l2"])
    A = core_probe[..., 0].permute(2, 1, 0).contiguous()   # (H, T, T)
    # reconstruction error against the real output
    o_recon = torch.einsum("htj,jhc->thc", A.float(), v[0].float())
    scale = cap["core"][0].abs().max().item() or 1.0
    rel_err = (o_recon - cap["core"][0].float()).abs().max().item() / scale
    return A.float().cpu().numpy(), rel_err


def _qwen3_5_hybrid_attention(model, tokenizer, input_ids) -> HybridAttentionData:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    GDN = mq.Qwen3_5GatedDeltaNet
    cap: dict[int, dict] = {}

    # wrap each Gated DeltaNet layer's chunk kernel to capture its exact inputs
    originals = []
    for idx, layer in enumerate(model.model.layers):
        mod = next((m for m in layer.modules() if isinstance(m, GDN)), None)
        if mod is None:
            continue

        def make(orig, idx):
            def cap_fn(query, key, value, **kw):
                out = orig(query, key, value, **kw)
                cap[idx] = dict(q=query.detach(), k=key.detach(), v=value.detach(),
                                g=kw["g"].detach(), beta=kw["beta"].detach(),
                                l2=kw.get("use_qk_l2norm_in_kernel", True),
                                core=out[0].detach())
                return out
            return cap_fn

        originals.append((mod, mod.chunk_gated_delta_rule))
        mod.chunk_gated_delta_rule = make(mod.chunk_gated_delta_rule, idx)

    try:
        with torch.no_grad():
            out = model(input_ids, output_attentions=True)
    finally:
        for mod, orig in originals:   # always restore
            mod.chunk_gated_delta_rule = orig

    # full-attention layers, in model order, from output_attentions
    layer_types = getattr(model.config, "layer_types", None)
    full_idx = ([i for i, t in enumerate(layer_types) if t == "full_attention"]
                if layer_types else [])
    full_attn = [a for a in (out.attentions or []) if a is not None]
    full_map = dict(zip(full_idx, full_attn))

    n_layers = len(model.model.layers)
    layers: list[LayerAttention] = []
    for i in range(n_layers):
        if i in cap:
            A, err = _reconstruct_gated_deltanet(cap[i], mq.torch_recurrent_gated_delta_rule)
            layers.append(LayerAttention(layer=i, kind="linear", attn=A, recon_error=err))
        elif i in full_map:
            A = full_map[i][0].to(torch.float32).cpu().numpy()   # (heads, seq, seq)
            layers.append(LayerAttention(layer=i, kind="full", attn=A))

    from .extract import _display_token
    tokens = [_display_token(t) for t in tokenizer.convert_ids_to_tokens(input_ids[0])]
    return HybridAttentionData(tokens=tokens, layers=layers,
                               model_name=getattr(model.config, "_name_or_path", ""))


# composite models expose the inner text config's type (…_text)
def hybrid_heatmap(data: HybridAttentionData, layer: int, head: int | None = None,
                   html_file: str | None = None):
    """Render one layer's effective-attention as a heatmap. Signed diverging
    colormap (linear-attention weights can be negative, unlike softmax). With
    ``head=None`` shows the mean over heads."""
    import html as _html

    la = next((l for l in data.layers if l.layer == layer), None)
    if la is None:
        raise ValueError(f"no layer {layer}")
    M = la.attn[head] if head is not None else la.attn.mean(0)
    T = M.shape[0]
    scale = float(np.abs(M).max()) or 1.0
    tag = (f"layer {layer} ({la.kind}"
           + (f", recon_err {la.recon_error:.1e}" if la.recon_error is not None else "")
           + f") · {'head ' + str(head) if head is not None else 'mean over heads'}")

    cell = 26
    pad_l, pad_t = 70, 90
    W, H = pad_l + T * cell + 10, pad_t + T * cell + 10
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             f'font-family="ui-monospace,Menlo,monospace" font-size="10">']
    parts.append(f'<text x="8" y="16" font-size="12" fill="#333">{_html.escape(tag)}</text>')
    parts.append(f'<text x="8" y="32" font-size="9" fill="#888">rows = query (from) · '
                 f'cols = key (into) · red +, blue − · |max| {scale:.3f}</text>')
    for j in range(T):  # column (key) labels, rotated
        x = pad_l + j * cell + cell / 2
        parts.append(f'<text x="{x}" y="{pad_t-6}" text-anchor="start" fill="#555" '
                     f'transform="rotate(-60 {x} {pad_t-6})">{_html.escape(data.tokens[j].strip()[:8])}</text>')
    for i in range(T):  # row (query) labels
        y = pad_t + i * cell + cell / 2 + 3
        parts.append(f'<text x="{pad_l-6}" y="{y}" text-anchor="end" fill="#555">'
                     f'{_html.escape(data.tokens[i].strip()[:9])}</text>')
        for j in range(T):
            v = float(M[i, j])
            a = max(-1.0, min(1.0, v / scale))
            color = (f"rgba(210,50,50,{0.08+0.85*a:.3f})" if a > 0
                     else (f"rgba(60,90,200,{0.08+0.85*-a:.3f})" if a < 0 else "#fff"))
            x = pad_l + j * cell
            yy = pad_t + i * cell
            parts.append(f'<rect x="{x}" y="{yy}" width="{cell-1}" height="{cell-1}" '
                         f'fill="{color}" stroke="#eee" stroke-width="0.5"><title>'
                         f'{data.tokens[i].strip()}→{data.tokens[j].strip()}: {v:+.4f}</title></rect>')
    parts.append("</svg>")
    svg = "".join(parts)

    if html_file:
        with open(html_file, "w") as f:
            f.write(f"<!DOCTYPE html><meta charset='utf-8'><body style='background:#fff'>{svg}</body>")
    try:
        from IPython.display import HTML, display
        display(HTML(svg))
    except ImportError:
        if not html_file:
            raise RuntimeError("not in IPython; pass html_file=") from None
    return svg


_HYBRID = {
    "qwen3_5": _qwen3_5_hybrid_attention,
    "qwen3_5_text": _qwen3_5_hybrid_attention,
}


def get_hybrid_attention(model, text: str, tokenizer=None, device: str | None = None,
                         dtype=None, max_tokens: int | None = None) -> HybridAttentionData:
    """Attention for a hybrid linear-attention model: real softmax for full
    layers, exact reconstructed effective-attention for linear layers.

    ``model`` is a model name (loaded eager so full-attn weights materialize) or
    a loaded model (pass ``tokenizer`` too).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if isinstance(model, str):
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(
            model, attn_implementation="eager", dtype=dtype or torch.float32)
    elif tokenizer is None:
        raise ValueError("pass a tokenizer when passing a loaded model")

    mtype = getattr(model.config, "model_type", None)
    if mtype not in _HYBRID:
        raise NotImplementedError(
            f"hybrid effective-attention not implemented for model_type {mtype!r}; "
            f"supported: {list(_HYBRID)}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()

    enc = tokenizer(text, return_tensors="pt", truncation=max_tokens is not None,
                    max_length=max_tokens)
    input_ids = enc["input_ids"].to(device)
    T = input_ids.shape[1]
    if T > 256:
        warnings.warn(f"sequence is {T} tokens; effective-attention reconstruction is "
                      f"O(T^2) per linear layer and may be slow/memory-heavy", stacklevel=2)
    return _HYBRID[mtype](model, tokenizer, input_ids)
