"""General hidden-state analysis: per-layer logit lens and cross-checkpoint
representation drift (CKA).

Two ways to *evaluate the residual stream* that complement ``probe`` (which
asks "is a label linearly decodable"):

- **logit lens** — unembed each layer's residual stream (after the model's final
  norm) to read the model's running next-token guess at every layer. No training;
  shows *when across depth* a prediction crystallizes.
- **CKA drift** — linear Centered Kernel Alignment between two checkpoints'
  hidden states on the same prompts, per layer. CKA≈1 means the representation is
  unchanged; lower means post-training reshaped it. This is the diff-two-
  checkpoints verb applied to representations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .attrib.models import LM


# ---- per-layer logit lens --------------------------------------------------
@dataclass
class LogitLens:
    tokens: list[str]
    layers: list[int]                       # model hidden_states indices used
    top_tokens: list[list[list[str]]]       # [layer][position] -> top-k token strings
    model_name: str = ""

    def trajectory(self, pos: int = -1) -> list[str]:
        """Top-1 next-token guess at ``pos`` as it evolves across layers."""
        return [self.top_tokens[li][pos][0] for li in range(len(self.layers))]


@torch.no_grad()
def logit_lens(lm: LM, text, layers: list[int] | None = None, top_k: int = 5,
               apply_final_norm: bool = True) -> LogitLens:
    """Project each layer's residual stream through the unembedding."""
    input_ids = lm.encode(text) if isinstance(text, str) else text.to(lm.device)
    out = lm.model(input_ids, output_hidden_states=True)
    hs = out.hidden_states                          # (L+1) of (1, seq, d)
    head = lm.model.get_output_embeddings()
    # the model's final norm, applied before unembed (standard logit-lens practice)
    norm = getattr(getattr(lm.model, "model", lm.model), "norm", None)

    idx = list(range(1, len(hs))) if layers is None else layers   # skip embeddings
    tokens = lm.token_strings(input_ids)
    top_tokens = []
    for li in idx:
        h = hs[li][0]
        if apply_final_norm and norm is not None:
            h = norm(h)
        logits = head(h).float()                    # (seq, vocab)
        top = logits.topk(top_k, dim=-1).indices    # (seq, top_k)
        per_pos = [[lm.tokenizer.decode([int(t)]).strip() for t in row] for row in top]
        top_tokens.append(per_pos)
    return LogitLens(tokens=tokens, layers=idx, top_tokens=top_tokens,
                     model_name=getattr(lm.model.config, "_name_or_path", ""))


# ---- CKA representation drift ----------------------------------------------
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019) between two activation matrices, each
    (n_samples, d). 1 = identical representation geometry, 0 = unrelated."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(Y.T @ X) ** 2
    xx = np.linalg.norm(X.T @ X)
    yy = np.linalg.norm(Y.T @ Y)
    return float(xy / (xx * yy)) if xx > 0 and yy > 0 else float("nan")


@dataclass
class CKADrift:
    layers: list[int]
    cka: np.ndarray                 # per-layer CKA (base vs other)
    name_base: str
    name_other: str
    n_vectors: int

    def most_changed(self, k: int = 3):
        order = np.argsort(self.cka)            # lowest CKA = most drift
        return [(self.layers[i], float(self.cka[i])) for i in order[:k]]


def cka_drift(lm_base: LM, lm_other: LM, texts: list[str],
              layers: list[int] | None = None, last_n: int | None = None) -> CKADrift:
    """Per-layer CKA between two checkpoints' hidden states on the same prompts.

    Both models must share a tokenizer (base and its fine-tune). ``last_n`` keeps
    only the last N positions of each prompt (default: all).
    """
    per_layer_base: dict[int, list[np.ndarray]] = {}
    per_layer_other: dict[int, list[np.ndarray]] = {}
    for text in texts:
        ids = lm_base.encode(text)
        hb = lm_base.hidden_states(ids, layers)
        ho = lm_other.hidden_states(ids.to(lm_other.device), layers)
        L = hb.shape[0]
        sel = slice(-last_n, None) if last_n else slice(None)
        for li in range(L):
            per_layer_base.setdefault(li, []).append(hb[li, sel].numpy())
            per_layer_other.setdefault(li, []).append(ho[li, sel].numpy())

    layer_idx = layers if layers is not None else list(range(len(per_layer_base)))
    ckas = []
    n_vec = 0
    for li in range(len(per_layer_base)):
        Xb = np.concatenate(per_layer_base[li], 0)
        Xo = np.concatenate(per_layer_other[li], 0)
        n_vec = Xb.shape[0]
        ckas.append(linear_cka(Xb, Xo))
    return CKADrift(layers=list(layer_idx), cka=np.array(ckas),
                    name_base=getattr(lm_base.model.config, "_name_or_path", "base"),
                    name_other=getattr(lm_other.model.config, "_name_or_path", "other"),
                    n_vectors=n_vec)


def cka_drift_chart(result: "CKADrift", html_file: str | None = None):
    """Bar chart of representation drift (1 - CKA) per layer; the most-reshaped
    layer is highlighted. Displays inline in a notebook and/or writes an SVG."""
    import html as _html

    layers = result.layers
    drift = [1.0 - float(c) for c in result.cka]
    n = len(layers)
    maxd = max(drift) or 1.0
    peak = int(np.argmax(drift))
    W, H, ml, mt, ph = 680, 280, 55, 44, 180
    pw = W - ml - 20
    title = f"representation drift by layer (1 - CKA):  {result.name_base}  →  {result.name_other}"
    p = [f'<svg width="100%" viewBox="0 0 {W} {H}" role="img" font-family="ui-monospace,Menlo,monospace">',
         f'<title>CKA drift</title><desc>per-layer representation drift between two checkpoints</desc>',
         f'<text x="20" y="22" font-size="13" fill="#222">{_html.escape(title)}</text>',
         f'<text x="20" y="38" font-size="10" fill="#888">higher = more reshaped by training · {result.n_vectors} vectors/layer</text>',
         f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#ccc"/>']
    for i, (l, d) in enumerate(zip(layers, drift)):
        x = ml + pw * (i + 0.15) / n
        bw = pw / n * 0.7
        h = ph * d / maxd
        color = "#d2643c" if i == peak else "#6b9bd1"
        p.append(f'<rect x="{x:.1f}" y="{mt+ph-h:.1f}" width="{bw:.1f}" height="{max(h,1):.1f}" fill="{color}" rx="2"><title>layer {l}: CKA={result.cka[i]:.4f}</title></rect>')
        p.append(f'<text x="{x+bw/2:.1f}" y="{mt+ph+14}" text-anchor="middle" font-size="9" fill="{"#222" if i==peak else "#888"}">L{l}</text>')
        if i == peak:
            p.append(f'<text x="{x+bw/2:.1f}" y="{mt+ph-h-5:.1f}" text-anchor="middle" font-size="10" font-weight="bold" fill="#222">{d:.3f}</text>')
    p.append("</svg>")
    svg = "".join(p)
    if html_file:
        with open(html_file, "w") as f:
            f.write(f"<!DOCTYPE html><meta charset='utf-8'><body style='background:#fff;margin:0'>"
                    f"<div style='width:680px;max-width:100%'>{svg}</div></body>")
    try:
        from IPython.display import HTML, display
        display(HTML(svg))
    except ImportError:
        if not html_file:
            raise RuntimeError("not in IPython; pass html_file=") from None
    return svg
