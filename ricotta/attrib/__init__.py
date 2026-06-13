from .attribute import Attribution, attribute, gradient_attribution, integrated_gradients, occlusion
from .diff import TokenDiff, attribution_diff, token_kl, token_logprob_diff
from .embed import (
    attribute_embeds,
    modality_contribution,
    text_to_embeds,
)
from .faithful import (
    agreement,
    comprehensiveness,
    deletion_curve,
    insertion_curve,
    sufficiency,
)
from .models import LM
from .relevance import build_judge_prompt, relevance_mask, top_t_positions
from .render import show_attribution, show_diff
from .spans import Span, SpanScore, aggregate, cot_faithfulness, span_ablation

__version__ = "0.2.0"
__all__ = [
    "LM",
    "attribute", "gradient_attribution", "integrated_gradients", "occlusion", "Attribution",
    "token_logprob_diff", "token_kl", "attribution_diff", "TokenDiff",
    "top_t_positions", "relevance_mask", "build_judge_prompt",
    "comprehensiveness", "sufficiency", "deletion_curve", "insertion_curve", "agreement",
    "show_diff", "show_attribution",
    "attribute_embeds", "modality_contribution", "text_to_embeds",
    "Span", "SpanScore", "aggregate", "span_ablation", "cot_faithfulness",
]
