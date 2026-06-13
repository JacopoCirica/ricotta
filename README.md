# ricotta

**Interpretability for LM post-training, at three altitudes.** One library to
answer *what did my fine-tune / RL run actually change?* — from the attention
patterns down to the internal features — unified by a single "diff two
checkpoints" verb.

| layer | question | module |
|---|---|---|
| **attention** | where does the model *look*? | `ricotta.attn` |
| **attribution** | which inputs *drive / changed* the output? | `ricotta.attrib` |
| **circuits** | which internal *features* rewired? | `ricotta.circuits` |

The first two need only `torch` + `transformers`. The circuits layer is an
optional extra (it builds on [circuit-tracer](https://github.com/decoderesearch/circuit-tracer)
and downloads per-model transcoders), imported lazily so everything else works
without it.

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

## What works where

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
