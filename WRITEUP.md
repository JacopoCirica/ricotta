# Exact effective-attention for hybrid linear-attention models

*Reconstructing interpretable attention maps for Qwen3.5's Gated DeltaNet layers.*

## Summary

Modern hybrid LMs (Qwen3.5, Qwen3-Next, and similar) replace most softmax
attention layers with **linear-attention** layers — Qwen3.5 uses Gated DeltaNet.
These layers have no attention matrix, so `output_attentions=True` silently
returns nothing for them and every attention-visualization tool goes blind on
~75% of the network. We show that an **exact** effective-attention matrix
nonetheless exists for these layers, give a method to recover it that requires
no closed-form derivation, validate it four ways, and report how the recovered
attention differs from the model's own softmax layers.

All numbers below are on **Qwen/Qwen3.5-4B**, fp32, 8 prompts, all 24 linear and
8 full-attention layers. Reproduce with `examples/effattn_validation.py`.

## Why an exact matrix exists

A Gated DeltaNet layer maintains a state `S_t ∈ R^{d_k × d_v}` and emits
`o_t = S_t^T q_t`. The recurrence (from the reference implementation) is

```
S_t = g_t · S_{t-1};   S_t += k_t ⊗ β_t (v_t − S_t^T k_t);   o_t = S_t^T q_t
```

with scalar gates `g_t` (decay) and `β_t`. The value vectors enter only as whole
columns scaled by scalars, and **value channels never mix**. Therefore each
output channel is

```
(o_t)_c = Σ_{j≤t} A[t,j] (v_j)_c      with the SAME scalar A[t,j] for every c,
```

i.e. an exact, per-head effective-attention matrix `A` exists — even though the
delta rule's `(I − β k kᵀ)` corrections make its closed form awkward.

## Recovering A without deriving it

Because `o` is **linear** in `v`, we recover `A` by re-running the layer's own
recurrence with indicator values: set `v = δ_j` (a one at position `j`, channel
0) and read column `j` off the output. Batching over `j` gives the full matrix
in one call. This reuses the model's exact kernel, so it is faithful by
construction — no reimplementation, no approximation.

## Validation

| check | result | meaning |
|---|---|---|
| **Cross-method agreement** | **6.98e-10** max diff | indicator-probe `A` vs an independent reverse-mode autograd Jacobian `∂oₜ/∂vⱼ` agree to ~1e-9 — two derivations corroborate |
| **Reconstruction at scale** | **7.9e-7** max, 2.8e-7 mean rel error (192 layer×prompt pairs) | `Σⱼ A[t,j] vⱼ` reproduces the real layer output to machine precision, everywhere |
| **Causality** | **exactly 0** mass above the diagonal | no future leakage |
| **Sanity on softmax** | measured neg-mass `0.0`, row-sum `1.000` | the same statistics on real softmax layers return their known exact values, confirming the measurement code |

## Finding: DeltaNet attention is signed and unnormalized

Measuring the recovered attention against the model's *own* softmax layers
(same model, same prompts):

| property | DeltaNet (linear) | softmax (full) |
|---|---|---|
| negative-mass fraction | **11.2 %** | 0 % |
| row sum `Σⱼ A[t,j]` | **0.013** | 1.000 |
| normalized entropy | 0.68 | 0.70 |
| locality (mean attended distance) | **1.94** | 2.77 |

Two qualitative differences stand out. **(1) ~11% of the attention mass is
negative** — the delta rule actively *subtracts* past values, something softmax
attention cannot express; these are not probabilities. **(2) Rows do not sum to
1** (≈0.01, vs exactly 1 for softmax) — the gated state is not a convex average,
so "effective attention" here is a signed read-weight, not a distribution.
Linear layers are also somewhat **more local** (attend ~1.9 vs ~2.8 tokens back)
at comparable entropy. Interpreting these maps as if they were softmax — reading
row-normalized probabilities — would be wrong; the sign and magnitude carry the
information.

## Limitations

- The short causal conv on q/k/v means `A` is over the **post-conv** values (the
  values the recurrence actually sees); the conv adds a separate local mixing
  over the original tokens, not modeled here.
- Single sequence, Qwen3.5 (`qwen3_5`) specifically; the registry generalizes to
  other gated-delta-rule models but each needs its capture point wired.
- This validates *fidelity* (the map exactly explains the layer output), not a
  causal-interpretability claim beyond that.

## Reproduce

```python
from ricotta import get_hybrid_attention
from ricotta.attn import hybrid_heatmap
data = get_hybrid_attention("Qwen/Qwen3.5-4B", "The capital of Texas is Austin.")
print(data)                    # 32 layers [8 full + 24 linear], max_recon_err ~1e-7
hybrid_heatmap(data, layer=30) # signed effective-attention; layer=3 for a softmax layer
```

Full validation: `python examples/effattn_validation.py` → `effattn_validation.json`.
