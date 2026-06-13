"""End-to-end: checkpoint -> attribution graph -> diff.

The trick for fine-tuned checkpoints: TransformerLens only knows base
architectures by name, so we load the fine-tune with HF transformers and pass
it via ``hf_model=`` while keeping the base architecture name — same approach
circuit-tracer's own gemma-2-2b-it demo relies on. The transcoders stay those
of the base model, which is what makes the two graphs comparable (and is an
assumption you must check: see the error-node influence in the report).
"""

from __future__ import annotations

import gc

import torch
from circuit_tracer import attribute
from circuit_tracer.graph import Graph
from circuit_tracer.replacement_model import ReplacementModel

QWEN3_TRANSCODERS = {
    "Qwen/Qwen3-0.6B": "mwhanna/qwen3-0.6b-transcoders-lowl0",
    "Qwen/Qwen3-1.7B": "mwhanna/qwen3-1.7b-transcoders-lowl0",
    "Qwen/Qwen3-4B": "mwhanna/qwen3-4b-transcoders",
    "Qwen/Qwen3-8B": "mwhanna/qwen3-8b-transcoders",
    "Qwen/Qwen3-14B": "mwhanna/qwen3-14b-transcoders-lowl0",
}


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(
    checkpoint: str,
    base_arch: str | None = None,
    transcoder_set: str | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    **model_kwargs,
) -> ReplacementModel:
    """Load any checkpoint of a supported base architecture as a ReplacementModel.

    checkpoint: HF repo or local path. If it *is* the base model, just pass it
    for both. Otherwise set base_arch to the architecture it was fine-tuned from.
    """
    base_arch = base_arch or checkpoint
    transcoder_set = transcoder_set or QWEN3_TRANSCODERS.get(base_arch)
    if transcoder_set is None:
        raise ValueError(
            f"no known transcoder set for {base_arch}; pass transcoder_set= explicitly"
        )
    device = device or default_device()

    # lazy_decoder reads decoder rows via safetensors get_slice with tensor
    # indices, which safetensors rejects ("Unsupported slice index") — load
    # decoders eagerly; bfloat16 keeps the qwen3 sets within ~10GB.
    kwargs: dict = {"lazy_decoder": False, **model_kwargs}
    if checkpoint != base_arch:
        from transformers import AutoModelForCausalLM

        kwargs["hf_model"] = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=dtype)

    return ReplacementModel.from_pretrained(
        base_arch, transcoder_set, device=device, dtype=dtype, **kwargs
    )


def attribute_checkpoint(
    checkpoint: str,
    prompt: str,
    base_arch: str | None = None,
    transcoder_set: str | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    graph_path: str | None = None,
    **attribute_kwargs,
) -> Graph:
    """Load a checkpoint, compute its attribution graph for prompt, free the model."""
    model = load_checkpoint(checkpoint, base_arch, transcoder_set, device, dtype)
    try:
        graph = attribute(prompt, model, **attribute_kwargs)
    finally:
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if graph_path:
        graph.to_pt(graph_path)
    return graph
