"""Validation suite for exact effective-attention reconstruction (Qwen3.5).

Runs four checks across several prompts and all 24 Gated DeltaNet layers:
  (1) cross-method agreement: indicator-probe A vs an INDEPENDENT autograd
      Jacobian do_t/dv_j (forward-linearity vs reverse-mode autodiff);
  (2) reconstruction fidelity at scale: rel error of  o == A @ v;
  (3) causality: mass above the diagonal;
  (4) descriptive findings: how DeltaNet effective-attention differs from the
      model's own softmax (full-attention) layers.
Writes effattn_validation.json.
"""
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

from ricotta.attn.hybrid import _reconstruct_gated_deltanet

MODEL = "Qwen/Qwen3.5-4B"
PROMPTS = [
    "The capital of the state containing Dallas is Austin.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "She picked up the red book and put it on the wooden shelf.",
    "If it rains tomorrow, the outdoor concert will be cancelled.",
    "The square root of one hundred and forty-four is twelve.",
    "Paris, the capital of France, sits on the river Seine.",
    "A group of wolves is called a pack, and they hunt together.",
    "The experiment failed because the temperature was far too high.",
]

tok = AutoTokenizer.from_pretrained(MODEL)
# eager attention so full-attention layers materialize softmax weights for the
# comparison (sdpa silently returns none with output_attentions=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float32, attn_implementation="eager").eval()
GDN = mq.Qwen3_5GatedDeltaNet
layer_types = model.config.layer_types
full_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]


def capture_and_run(input_ids):
    cap = {}
    saved = []
    for idx, layer in enumerate(model.model.layers):
        mod = next((m for m in layer.modules() if isinstance(m, GDN)), None)
        if mod is None:
            continue
        def make(orig, idx):
            def fn(query, key, value, **kw):
                out = orig(query, key, value, **kw)
                cap[idx] = dict(q=query.detach(), k=key.detach(), v=value.detach(),
                                g=kw["g"].detach(), beta=kw["beta"].detach(),
                                l2=kw.get("use_qk_l2norm_in_kernel", True), core=out[0].detach())
                return out
            return fn
        saved.append((mod, mod.chunk_gated_delta_rule))
        mod.chunk_gated_delta_rule = make(mod.chunk_gated_delta_rule, idx)
    try:
        with torch.no_grad():
            out = model(input_ids, output_attentions=True)
    finally:
        for mod, orig in saved:
            mod.chunk_gated_delta_rule = orig
    return cap, out


def jacobian_A(cap, head):
    """Independent A[head] via reverse-mode autodiff of o[:,head,0] wrt v[:,head,0]."""
    q, k, v, g, beta = cap["q"], cap["k"], cap["v"], cap["g"], cap["beta"]
    T = v.shape[1]
    base = v[0, :, head, 0].clone()
    def f(vc):
        vv = v.clone()
        vv[0, :, head, 0] = vc
        o, _ = mq.torch_recurrent_gated_delta_rule(
            q, k, vv, g=g, beta=beta, initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=cap["l2"])
        return o[0, :, head, 0]
    return torch.autograd.functional.jacobian(f, base).numpy()   # (T,T) = A[head]


def stats(A):
    """A: (H,T,T) lower-triangular. Descriptive properties, averaged over heads
    and query positions t>=1."""
    H, T, _ = A.shape
    neg, rowsum, ent, loc, n = 0.0, [], [], [], 0
    tot_abs, tot_neg = 0.0, 0.0
    for h in range(H):
        for t in range(1, T):
            row = A[h, t, : t + 1]
            a = np.abs(row)
            s = a.sum()
            if s <= 0:
                continue
            tot_abs += s
            tot_neg += a[row < 0].sum()
            rowsum.append(row.sum())
            p = a / s
            ent.append(-(p * np.log(p + 1e-12)).sum() / np.log(t + 1 + 1e-9) if t >= 1 else 0)
            dist = np.arange(t, -1, -1)            # t-j for j=0..t
            loc.append(float((p * dist).sum()))
            n += 1
    return dict(neg_mass_frac=tot_neg / (tot_abs + 1e-12),
                rowsum_mean=float(np.mean(rowsum)), rowsum_std=float(np.std(rowsum)),
                norm_entropy=float(np.mean(ent)), locality=float(np.mean(loc)), n=n)


# ---- run ----
recon_errs, causal_max = [], []
cross = []
lin_stats = {i: [] for i in range(len(model.model.layers)) if layer_types[i] == "linear_attention"}
full_stats = {i: [] for i in full_idx}

for pi, text in enumerate(PROMPTS):
    ids = tok(text, return_tensors="pt").input_ids
    cap, out = capture_and_run(ids)
    full_attn = [a for a in (out.attentions or []) if a is not None]
    full_map = dict(zip(full_idx, full_attn))

    for li in sorted(cap):
        A, err = _reconstruct_gated_deltanet(cap[li], mq.torch_recurrent_gated_delta_rule)
        recon_errs.append(err)
        causal_max.append(float(np.abs(np.triu(A, 1)).max()))
        lin_stats[li].append(stats(A))
        # cross-method check on first prompt, first linear layer, head 0
        if pi == 0 and li == min(cap):
            Aj = jacobian_A(cap[li], head=0)
            cross.append(float(np.abs(A[0] - Aj).max()))

    for li, a in full_map.items():
        full_stats[li].append(stats(a[0].numpy()))

def agg(d):
    keys = ["neg_mass_frac", "rowsum_mean", "norm_entropy", "locality"]
    vals = {k: float(np.mean([s[k] for layer in d.values() for s in layer])) for k in keys}
    return vals

result = {
    "model": MODEL, "n_prompts": len(PROMPTS),
    "n_linear_layers": len(lin_stats), "n_full_layers": len(full_stats),
    "cross_method_max_diff": max(cross) if cross else None,
    "reconstruction": {"max_rel_err": float(np.max(recon_errs)),
                       "mean_rel_err": float(np.mean(recon_errs)),
                       "n_layer_prompt_pairs": len(recon_errs)},
    "causality_max_above_diag": float(np.max(causal_max)),
    "linear_attention": agg(lin_stats),
    "full_attention": agg(full_stats),
}
print(json.dumps(result, indent=2))
with open("effattn_validation.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nwrote effattn_validation.json")
