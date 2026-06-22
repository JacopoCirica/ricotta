# ricotta

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JacopoCirica/ricotta/blob/main/examples/ricotta_colab_demo.ipynb)

**Interpretability for LM post-training.** One library to answer *what did my
fine-tune / RL run actually change?* — from attention patterns down to internal
features and chain-of-thought faithfulness — unified by a single "diff two
checkpoints" verb.

| module | question it answers |
|---|---|
| `ricotta.attn` | where does the model *look*? (incl. exact effective-attention for hybrid linear-attention layers, e.g. Qwen3.5) |
| `ricotta.attrib` | which input tokens *drive / changed* the output? + per-step CoT faithfulness, modality |
| `ricotta.tts` | which reasoning steps are *causally load-bearing*? (True Thinking Score, [arXiv:2510.24941](https://arxiv.org/abs/2510.24941)) |
| `ricotta.monitor` | does the answer *depend on the visible CoT*? is it still monitorable after training? ([arXiv:2507.11473](https://arxiv.org/abs/2507.11473)) |
| `ricotta.probe` | *when/where* is the answer linearly decodable? (belief probing) |
| `ricotta.hidden` | per-layer logit lens + cross-checkpoint representation drift (CKA) |
| `ricotta.circuits` | which internal *features* rewired? (attribution graphs) |

Everything but `circuits` needs only `torch` + `transformers`; `tts`/`monitor`
work on any HF reasoning model (Qwen3.5 included). `circuits` is an optional
extra (it builds on [circuit-tracer](https://github.com/decoderesearch/circuit-tracer)
and downloads per-model transcoders), imported lazily so the rest works without
it. `probe` needs the `[probe]` extra (scikit-learn).

## Install

The distribution is **`ricotta-interp`**; the import name is **`ricotta`**
(the bare `ricotta` on PyPI is held by an abandoned 2017 package):

```bash
pip install ricotta-interp              # attn + attrib  (torch + transformers)
pip install "ricotta-interp[circuits]"  # + feature-circuit layer (circuit-tracer)
```

```python
import ricotta   # import name stays `ricotta`
```

Or straight from GitHub:

```bash
pip install git+https://github.com/JacopoCirica/ricotta.git
pip install "ricotta-interp[circuits] @ git+https://github.com/JacopoCirica/ricotta.git"
```

## The headline: one post-training report

```python
from ricotta import posttrain_report

md = posttrain_report(
    base="Qwen/Qwen3-4B",
    checkpoint="/path/to/sdpo-checkpoint",
    prompts=my_probe_prompts,
    circuits=True,            # add the feature-circuit layer (needs the extra)
    out_dir="report/",        # writes heatmaps + report.md
)
print(md)
```

Drop it into your eval loop to track what each checkpoint does, mechanistically.

## Or use the layers directly

```python
# attention — where it looks
from ricotta import get_attention, head_view
head_view(get_attention("Qwen/Qwen3-0.6B", "The cat sat on the mat."))

# attribution — what drives the output, checkpoint diffing, CoT, modalities
from ricotta import LM, token_logprob_diff, show_diff, cot_faithfulness, attribute_embeds
base, ft = LM.load("Qwen/Qwen3-0.6B"), LM.load("Qwen/Qwen3Guard-Gen-0.6B")
show_diff(token_logprob_diff(base, ft, base.encode("The capital of Texas is Austin.")))

# circuits — which features rewired (needs the `circuits` extra)
from ricotta import attribute_checkpoint, diff_graphs, to_markdown
ga = attribute_checkpoint("Qwen/Qwen3-0.6B", prompt)
gb = attribute_checkpoint("/path/to/ft", prompt, base_arch="Qwen/Qwen3-0.6B")
print(to_markdown(diff_graphs(ga, gb)))
```

Each layer's full API is documented in its subpackage; see `examples/`.

## Hybrid linear-attention models (Qwen3.5)

Models like Qwen3.5 interleave a few softmax layers with many **Gated DeltaNet**
linear-attention layers, which have no softmax matrix — so `output_attentions`
silently drops them. ricotta reconstructs their **exact effective attention**:
because a linear layer's output is a linear function of its values
(`o_t = Σ_{j≤t} A[t,j] v_j`), we recover `A` by re-running the layer's own
recurrence with indicator values — no approximation. It's verified by
reconstruction (`Σ_j A[t,j] v_j` reproduces the real output to ~1e-7).

```python
from ricotta import get_hybrid_attention
from ricotta.attn import hybrid_heatmap

data = get_hybrid_attention("Qwen/Qwen3.5-4B", "The capital of Texas is Austin.")
print(data)        # 32 layers [8 full + 24 linear], max_recon_err ~1e-7
hybrid_heatmap(data, layer=30)   # signed effective-attention heatmap
```

Full-attention layers keep their real softmax weights at their true indices
(e.g. 3, 7, …, 31); linear layers are reconstructed and **signed** (the delta
rule subtracts as well as adds — rows are not a probability distribution).
Needs `transformers >= 4.58` (so do *not* combine with the `circuits` extra,
which pins older transformers — and circuits can't analyze Qwen3.5 anyway).

## Hidden-state analysis (`ricotta.probe`, `ricotta.hidden`)

Three ways to *evaluate the residual stream*, complementing attribution:

```python
from ricotta import LM, logit_lens, cka_drift, run_belief_experiment, Question

lm = LM.load("Qwen/Qwen3-0.6B")

# 1. belief probing — when/where is the answer linearly decodable? (needs [probe])
res = run_belief_experiment(lm, questions, n_positions=30, cache_path="acts.npz")
res.peak()          # (layer, offset, accuracy)

# 2. per-layer logit lens — the model's running next-token guess across depth
ll = logit_lens(lm, "The capital of France is")
ll.trajectory(pos=-1)     # top-1 guess at each layer

# 3. cross-checkpoint representation drift — did post-training reshape the geometry?
d = cka_drift(LM.load("Qwen/Qwen3-0.6B"), LM.load("path/to/sdpo"), prompts)
d.most_changed()    # layers with lowest CKA = most drift
```

`hidden_states(text)` on the `LM` wrapper returns the raw residual stream
`(layers, seq, d)`. **Belief probing reproduces** the method of
[Exploring belief states in LLM CoT](https://www.lesswrong.com/posts/ncpdXznDMxDZDyn6J);
on a *reasoning-free* task (numeric comparison) the answer is decodable from the
start rather than emerging late — an honest reminder that belief dynamics are
task-dependent (see `examples/belief_probe_demo.py`). CKA drift is the
diff-two-checkpoints verb applied to representations.

## Chain-of-thought faithfulness & monitorability (`ricotta.tts`, `ricotta.monitor`)

Is the model's reasoning real, or decorative — and does training erode that?

**`tts` — True Thinking Score** ([arXiv:2510.24941](https://arxiv.org/abs/2510.24941)):
the causal contribution of each reasoning step. Perturb numbers in a step and
measure how `P(correct answer)` changes (necessity + sufficiency). Steps ≥0.7
are *true-thinking*, ≤0.005 *decorative*.

```python
from ricotta import LM, true_thinking_score, tts_chart
from ricotta.tts import generate_reasoning

lm = LM.load("Qwen/Qwen3.5-4B")
prompt, reasoning = generate_reasoning(lm, "A store has 33 red, 19 blue, 14 green marbles; "
                                           "it sells 6, 4, 2. How many remain?")
res = true_thinking_score(lm, prompt, reasoning, answer="54")
tts_chart(res)        # green = true-thinking, gray = decorative
```

**`monitor` — CoT monitorability** ([arXiv:2507.11473](https://arxiv.org/abs/2507.11473)):
the *latent-reasoning gap* = `P(correct | full CoT) − P(correct | empty think)`.
Large = the answer depends on the visible reasoning (monitorable); ~0 =
decorative. Aggregate with a bootstrap CI and **diff across checkpoints** to ask
*did RL training make the CoT less load-bearing?* — the measurement the
monitorability paper calls for.

```python
from ricotta import monitorability_over_dataset, monitorability_diff, monitorability_chart

problems = [("…question…", "answer"), ...]          # e.g. GSM8K / MATH
base = monitorability_over_dataset(lm_base, problems)
ft   = monitorability_over_dataset(lm_ft,   problems)
d = monitorability_diff(base, ft)
print(d, d.significant())                            # is the change real?
monitorability_chart(base, ft)
```

## What works where

- **tts / monitor** — any HF reasoning model that emits a CoT; need a known
  answer for the math/reasoning tasks they score (Qwen3.5 works).
- **attn** — any HF model with eager attention (GQA fine).
- **attrib** — any decoder-only HF causal LM; no transcoders needed, so it's the
  one layer that runs on hybrid-attention models (e.g. Qwen3.5). Includes
  gradient / IG / occlusion attribution, checkpoint diffing (logprob-diff, KL,
  attribution-diff), relevance masking, **CoT span-faithfulness**, **projected-
  modality attribution**, and a faithfulness eval harness.
- **circuits** — only base architectures with public transcoders (Qwen3,
  Gemma-2/3, Llama-3.x, GPT-OSS) and their fine-tunes.

Run these against model **weights** on hardware you control — not a serving
endpoint (which exposes outputs, not internals). See `ACKNOWLEDGEMENTS.md` for
credits and licensing of the circuits layer's dependencies.

## License

MIT (this code). The circuits layer depends on circuit-tracer (MIT, not
vendored) and downloads third-party transcoders at runtime that carry their own
licenses — see `ACKNOWLEDGEMENTS.md`.
