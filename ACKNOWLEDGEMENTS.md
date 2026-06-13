# Acknowledgements

ricotta's **circuits** layer is a thin tool built on top of, and requires:

- **[circuit-tracer](https://github.com/decoderesearch/circuit-tracer)** (MIT) —
  the attribution-graph engine. ricotta depends on it as a library; no
  circuit-tracer code is vendored here. It implements the methods of
  Ameisen et al. (2025) and Lindsey et al. (2025),
  https://transformer-circuits.pub/2025/attribution-graphs/methods.html

- **Pretrained transcoders**, downloaded from the Hugging Face Hub at runtime
  and **not redistributed** here — each carries its own license:
  - Qwen3 transcoders by `mwhanna`
  - Gemma-2 / Gemma-3 GemmaScope transcoders (Google) and `mntss` ports
  - Llama transcoders by `mntss`, Llama-3.1 by Meta (`facebook/crv-*`)

The **attn** and **attrib** layers (attention visualization and input-token
attribution) are original code with no third-party runtime dependencies beyond
`torch` and `transformers`.

attention visualization is inspired by [BertViz](https://github.com/jessevig/bertviz);
the contrastive token-diff view is inspired by relevance-masked self-distillation.
