"""Transcoder-transfer validation.

Circuit diffing applies *base-model* transcoders to a fine-tuned checkpoint.
That's only sound if the transcoders still reconstruct the fine-tune's MLP
outputs well. This module measures per-layer FVU (fraction of variance
unexplained) on a set of prompts: FVU ~ 0 is perfect reconstruction, FVU >> a
few times the base model's own FVU means the diff is untrustworthy.
"""

from __future__ import annotations

import torch
from circuit_tracer.replacement_model import ReplacementModel


@torch.no_grad()
def transcoder_fvu(model: ReplacementModel, prompts: list[str]) -> torch.Tensor:
    """Per-layer FVU of the model's transcoders, averaged over prompts."""
    n_layers = model.cfg.n_layers
    sse = torch.zeros(n_layers)
    svar = torch.zeros(n_layers)

    # exact hook names: a substring filter would also match the wrapped inner
    # MLP's hooks (blocks.N.mlp.old_mlp.hook_in) and double the cache
    in_names = [f"blocks.{l}.{model.feature_input_hook}" for l in range(n_layers)]
    out_names = [f"blocks.{l}.{model.original_feature_output_hook}" for l in range(n_layers)]

    for prompt in prompts:
        tokens = model.ensure_tokenized(prompt)

        mlp_in_cache, in_hooks, _ = model.get_caching_hooks(lambda name: name in in_names)
        mlp_out_cache, out_hooks, _ = model.get_caching_hooks(lambda name: name in out_names)
        model.run_with_hooks(tokens, fwd_hooks=in_hooks + out_hooks)

        mlp_in = torch.cat([mlp_in_cache[n] for n in in_names], dim=0)     # (layers, pos, d)
        mlp_out = torch.cat([mlp_out_cache[n] for n in out_names], dim=0)  # (layers, pos, d)

        components = model.transcoders.compute_attribution_components(
            mlp_in, model.zero_positions
        )
        recon = components["reconstruction"]

        # skip the BOS position(s) the library also zeroes out
        keep = slice(model.zero_positions.stop, None)
        err = (mlp_out[:, keep] - recon[:, keep]).float()
        tgt = mlp_out[:, keep].float()
        sse += err.pow(2).sum(dim=(1, 2)).cpu()
        svar += (tgt - tgt.mean(dim=1, keepdim=True)).pow(2).sum(dim=(1, 2)).cpu()

    return sse / svar


def compare_fvu(
    fvu_base: torch.Tensor, fvu_ft: torch.Tensor, ratio_threshold: float = 2.0
) -> dict:
    """Summarize transfer quality. verdict 'ok' if the fine-tune's FVU stays
    within ratio_threshold x the base FVU on (almost) every layer."""
    ratio = fvu_ft / fvu_base
    return {
        "fvu_base_mean": float(fvu_base.mean()),
        "fvu_finetuned_mean": float(fvu_ft.mean()),
        "worst_layer": int(ratio.argmax()),
        "worst_ratio": float(ratio.max()),
        "median_ratio": float(ratio.median()),
        "verdict": "ok" if float(ratio.median()) < ratio_threshold else "poor transfer",
    }
