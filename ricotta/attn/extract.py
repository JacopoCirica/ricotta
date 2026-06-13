"""Attention extraction from Hugging Face causal language models.

Works with any decoder-only model whose forward pass returns per-layer
attention weights of shape [batch, num_heads, seq, seq] when called with
``output_attentions=True`` — Qwen3, Llama, Mistral, GPT-2, etc. GQA models
are transparent here: HF returns weights at query-head granularity.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class AttentionData:
    """Attention weights for one input sequence.

    attentions: float32 array [num_layers, num_heads, seq, seq];
    each row attentions[l, h, q, :] sums to ~1 (softmax output).
    """

    tokens: list[str]
    attentions: np.ndarray
    model_name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.attentions.ndim != 4:
            raise ValueError(
                f"attentions must be [layers, heads, seq, seq], got shape {self.attentions.shape}"
            )
        if self.attentions.shape[2] != len(self.tokens) or self.attentions.shape[3] != len(self.tokens):
            raise ValueError(
                f"attention seq dims {self.attentions.shape[2:]} do not match {len(self.tokens)} tokens"
            )

    @property
    def num_layers(self) -> int:
        return self.attentions.shape[0]

    @property
    def num_heads(self) -> int:
        return self.attentions.shape[1]

    @property
    def seq_len(self) -> int:
        return self.attentions.shape[2]

    def __repr__(self) -> str:
        return (
            f"AttentionData(model={self.model_name!r}, layers={self.num_layers}, "
            f"heads={self.num_heads}, seq_len={self.seq_len})"
        )


def _display_token(tok: str) -> str:
    """Make subword tokens readable: BPE/SentencePiece space markers -> real
    leading space, newlines -> visible symbol."""
    tok = tok.replace("Ġ", " ").replace("▁", " ")
    tok = tok.replace("Ċ", "\\n").replace("\n", "\\n").replace("\t", "\\t")
    return tok


def get_attention(
    model,
    text: str,
    tokenizer=None,
    device: str | None = None,
    max_tokens: int | None = None,
    dtype: torch.dtype | None = None,
) -> AttentionData:
    """Run a forward pass and collect attention weights for ``text``.

    ``model`` is either a model name string (loaded with eager attention so
    weights are materialized) or an already-loaded HF model, in which case
    ``tokenizer`` must be given too.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if isinstance(model, str):
        model_name = model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            dtype=dtype or torch.float32,
        )
    else:
        if tokenizer is None:
            raise ValueError("pass a tokenizer when passing a loaded model")
        model_name = getattr(model.config, "_name_or_path", model.__class__.__name__)
        impl = getattr(model.config, "_attn_implementation", None)
        if impl not in (None, "eager"):
            warnings.warn(
                f"model uses attn_implementation={impl!r}; transformers will fall back "
                "to eager to materialize attention weights, which may be slow. "
                "Load with attn_implementation='eager' to silence this.",
                stacklevel=2,
            )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()

    enc = tokenizer(text, return_tensors="pt", truncation=max_tokens is not None, max_length=max_tokens)
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] == 0:
        raise ValueError("text tokenized to zero tokens")

    with torch.no_grad():
        out = model(input_ids, output_attentions=True)

    if out.attentions is None or out.attentions[0] is None:
        raise RuntimeError(
            "model returned no attention weights; reload it with attn_implementation='eager'"
        )

    # tuple of num_layers tensors [1, heads, seq, seq] -> [layers, heads, seq, seq]
    attn = torch.stack([a[0] for a in out.attentions]).to(torch.float32).cpu().numpy()

    tokens = [_display_token(t) for t in tokenizer.convert_ids_to_tokens(input_ids[0])]
    return AttentionData(tokens=tokens, attentions=attn, model_name=model_name)


def from_attentions(attentions, tokens: list[str], model_name: str = "") -> AttentionData:
    """Build AttentionData from precomputed weights.

    ``attentions`` is whatever your forward pass produced: a tuple/list of
    per-layer tensors [batch, heads, seq, seq] (batch index 0 is used), or a
    single stacked array [layers, heads, seq, seq].
    """
    if isinstance(attentions, (tuple, list)):
        layers = []
        for a in attentions:
            if isinstance(a, torch.Tensor):
                a = a.detach().to(torch.float32).cpu().numpy()
            layers.append(a[0] if a.ndim == 4 else a)
        attn = np.stack(layers)
    else:
        attn = attentions
        if isinstance(attn, torch.Tensor):
            attn = attn.detach().to(torch.float32).cpu().numpy()
    return AttentionData(tokens=list(tokens), attentions=attn.astype(np.float32), model_name=model_name)
