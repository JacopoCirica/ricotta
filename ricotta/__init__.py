"""ricotta — interpretability for LM post-training, at three altitudes.

- ``ricotta.attn``    — attention visualization (where the model looks)
- ``ricotta.attrib``  — input-token attribution (what drives / changed the output)
- ``ricotta.circuits``— feature-circuit diffing (which internal features rewired)

The first two need only ``torch`` + ``transformers``. The circuits layer needs
the optional ``circuits`` extra (circuit-tracer + per-model transcoders); it is
imported lazily so the rest of ricotta works without it.
"""

from . import attn, attrib
from .attn import (
    AttentionData,
    HybridAttentionData,
    LayerAttention,
    from_attentions,
    get_attention,
    get_hybrid_attention,
    head_view,
)
from .attrib import (
    LM,
    Attribution,
    Span,
    SpanScore,
    aggregate,
    agreement,
    attribute,
    attribute_embeds,
    attribution_diff,
    build_judge_prompt,
    comprehensiveness,
    cot_faithfulness,
    deletion_curve,
    gradient_attribution,
    insertion_curve,
    integrated_gradients,
    modality_contribution,
    occlusion,
    relevance_mask,
    show_attribution,
    show_diff,
    span_ablation,
    sufficiency,
    text_to_embeds,
    token_kl,
    token_logprob_diff,
    top_t_positions,
    TokenDiff,
)

HAS_CIRCUITS = False
try:  # optional: requires the `circuits` extra (circuit-tracer + transcoders)
    from . import circuits
    from .circuits import (
        GraphDiff,
        QWEN3_TRANSCODERS,
        attribute_checkpoint,
        diff_graphs,
        load_checkpoint,
        to_markdown,
    )

    HAS_CIRCUITS = True
except ImportError:  # circuit-tracer not installed
    circuits = None

from . import hidden, monitor, probe, tts
from .attrib import cot_faithfulness_chart
from .monitor import (
    GapResult,
    MonitorabilityDiff,
    MonitorabilitySummary,
    latent_reasoning_gap,
    monitorability_chart,
    monitorability_diff,
    monitorability_over_dataset,
)
from .tts import TTSResult, TTSStep, true_thinking_score, tts_chart
from .hidden import CKADrift, LogitLens, cka_drift, cka_drift_chart, linear_cka, logit_lens
from .probe import BeliefResult, Question, belief_curve, run_belief_experiment
from .report import posttrain_report

__version__ = "0.1.0"
__all__ = [
    "attn", "attrib", "circuits", "HAS_CIRCUITS",
    # attn
    "AttentionData", "get_attention", "from_attentions", "head_view",
    "get_hybrid_attention", "HybridAttentionData", "LayerAttention",
    # attrib
    "LM", "attribute", "gradient_attribution", "integrated_gradients", "occlusion", "Attribution",
    "token_logprob_diff", "token_kl", "attribution_diff", "TokenDiff",
    "top_t_positions", "relevance_mask", "build_judge_prompt",
    "comprehensiveness", "sufficiency", "deletion_curve", "insertion_curve", "agreement",
    "show_diff", "show_attribution",
    "attribute_embeds", "modality_contribution", "text_to_embeds",
    "Span", "SpanScore", "aggregate", "span_ablation", "cot_faithfulness",
    # circuits (present only with the extra)
    "diff_graphs", "to_markdown", "attribute_checkpoint", "load_checkpoint",
    "QWEN3_TRANSCODERS", "GraphDiff",
    # probe (belief-state probing; needs the `probe` extra)
    "probe", "Question", "run_belief_experiment", "BeliefResult", "belief_curve",
    # tts — True Thinking Score (causal CoT step contribution)
    "tts", "true_thinking_score", "TTSResult", "TTSStep", "tts_chart",
    # monitor — CoT monitorability (latent-reasoning gap, checkpoint diff)
    "monitor", "latent_reasoning_gap", "GapResult", "monitorability_over_dataset",
    "MonitorabilitySummary", "monitorability_diff", "MonitorabilityDiff", "monitorability_chart",
    # hidden-state analysis: per-layer logit lens + CKA representation drift
    "hidden", "logit_lens", "LogitLens", "cka_drift", "CKADrift", "linear_cka",
    "cka_drift_chart", "cot_faithfulness_chart",
    # unified
    "posttrain_report",
]
