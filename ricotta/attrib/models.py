"""Thin wrapper around a HF causal LM for attribution.

Holds the model + tokenizer + device and exposes the few primitives the rest of
the library needs: tokenization, per-token log-probabilities of the actual
continuation, and the full next-token log-prob distribution. Everything is
decoder-only and single-sequence (batch dim 1) by design.
"""

from __future__ import annotations

import torch


def pick_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LM:
    def __init__(self, model, tokenizer, device: torch.device):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def load(
        cls,
        name: str,
        device=None,
        dtype: torch.dtype | None = None,
        eager_attention: bool = False,
        gradient_checkpointing: bool = False,
    ) -> "LM":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = pick_device(device)
        tok = AutoTokenizer.from_pretrained(name)
        kwargs = {"dtype": dtype or torch.float32}
        if eager_attention:
            kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        return cls(model, tok, device)

    # ---- tokenization ------------------------------------------------------
    def encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=add_special_tokens)
        return ids["input_ids"].to(self.device)

    def token_strings(self, input_ids: torch.Tensor) -> list[str]:
        toks = self.tokenizer.convert_ids_to_tokens(input_ids.flatten().tolist())
        return [t.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\\n") for t in toks]

    @property
    def baseline_token_id(self) -> int:
        for cand in (self.tokenizer.pad_token_id, self.tokenizer.eos_token_id,
                     self.tokenizer.bos_token_id):
            if cand is not None:
                return cand
        return 0

    # ---- scoring -----------------------------------------------------------
    @torch.no_grad()
    def token_logprobs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Log p(x_i | x_<i) for the *actual* tokens. Shape (seq,); index 0 is
        NaN (nothing predicts the first token)."""
        logits = self.model(input_ids).logits[0]              # (seq, vocab)
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        seq = input_ids.shape[1]
        out = torch.full((seq,), float("nan"), device=self.device)
        # logits[i-1] predicts token at position i
        tgt = input_ids[0, 1:]                                 # (seq-1,)
        out[1:] = logprobs[:-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return out

    @torch.no_grad()
    def next_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full logits, shape (seq, vocab); row i predicts token i+1."""
        return self.model(input_ids).logits[0].float()

    @torch.no_grad()
    def hidden_states(self, text, layers: list[int] | None = None) -> torch.Tensor:
        """Residual-stream activations for ``text`` (str or input_ids).

        Returns float32 (n_selected_layers, seq, d_model) on CPU. Index 0 of the
        model's hidden_states is the embedding layer; layers 1..L are block
        outputs. ``layers`` selects a subset (default: all, including embeddings).
        """
        input_ids = self.encode(text) if isinstance(text, str) else text.to(self.device)
        out = self.model(input_ids, output_hidden_states=True)
        hs = out.hidden_states                                  # tuple (L+1) of (1, seq, d)
        idx = range(len(hs)) if layers is None else layers
        return torch.stack([hs[i][0].float().cpu() for i in idx])
