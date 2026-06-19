"""True Thinking Score (TTS) — causal contribution of each reasoning step.

Implements the metric of "Measuring the True Reasoning of LLMs"
(arXiv:2510.24941) / the Thinking Machines recipe, locally on a HF model.

For each reasoning step we measure P(correct answer | reasoning prefix) under a
2×2 of {context intact/perturbed} × {step intact/perturbed}, scored by early
exit (close </think>, append \\boxed{answer}). Perturbation adds small integer
offsets to numbers in a step (or removes it if non-numeric) — a sharper
counterfactual than blunt token ablation. Combine as

    necessity   = |S(intact ctx, intact step)  − S(intact ctx, perturbed step)|
    sufficiency = |S(perturbed ctx, intact step) − S(perturbed ctx, perturbed step)|
    TTS(step)   = ½ (necessity + sufficiency)

Steps with TTS ≥ 0.7 are "true-thinking" (causally load-bearing); ≤ 0.005 are
"decorative". This is the rigorous version of ``cot_faithfulness``.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

import torch

from .attrib.models import LM

THINK_OPEN, THINK_CLOSE = "<think>\n", "\n</think>\n\n"
ANSWER_TEMPLATE = "The answer is \\boxed{"
OFFSETS = [-3, -2, -1, 1, 2, 3]
_NUM = re.compile(r"-?\d+")
_SELF_VERIFY = re.compile(r"(?i)\b(re-?check|verif|double[- ]?check|wait,|let me (re|check|make)|make sure)\b")
# discourse markers that tend to open a new reasoning step
_MARKERS = (r"First|Second|Third|Next|Then|Now|So|Therefore|Thus|Hence|Finally|"
            r"Step\s*\d|We\s|Let\s|Since|Because|However|Also|Alternatively")


@dataclass
class TTSStep:
    index: int
    text: str
    necessity: float
    sufficiency: float
    tts: float
    scores: dict                       # S11, S10, S01, S00
    label: str                         # 'true-thinking' | 'decorative' | 'intermediate'
    self_verification: bool = False

    def __repr__(self):
        sv = " [self-verify]" if self.self_verification else ""
        return f"TTSStep({self.index}: tts={self.tts:.3f} {self.label}{sv} | {self.text[:48]!r})"


@dataclass
class TTSResult:
    problem: str
    answer: str
    steps: list[TTSStep]
    model_name: str = ""
    meta: dict = field(default_factory=dict)

    def true_thinking(self):
        return [s for s in self.steps if s.label == "true-thinking"]

    def decorative(self):
        return [s for s in self.steps if s.label == "decorative"]

    def __repr__(self):
        return (f"TTSResult(n_steps={len(self.steps)}, "
                f"true-thinking={len(self.true_thinking())}, decorative={len(self.decorative())})")


# ---- segmentation & perturbation -------------------------------------------
def segment_steps(reasoning: str) -> list[str]:
    """Split a reasoning trace into steps on paragraph breaks and discourse
    markers (not bare sentence boundaries)."""
    chunks = [c for c in re.split(r"\n\s*\n", reasoning.strip()) if c.strip()]
    steps = []
    for c in chunks:
        # further split a long chunk before discourse markers
        parts = re.split(rf"(?<=[.\n])\s+(?=(?:{_MARKERS})\b)", c.strip())
        steps.extend(p.strip() for p in parts if p.strip())
    return steps or ([reasoning.strip()] if reasoning.strip() else [])


def perturb_step(text: str, rng: random.Random) -> str | None:
    """Add a nonzero integer offset (in {-3..3}) to each number in the step.
    Returns None if the step has no numbers (signal: remove it)."""
    if not _NUM.search(text):
        return None
    out, last = [], 0
    for m in _NUM.finditer(text):
        out.append(text[last:m.start()])
        out.append(str(int(m.group()) + rng.choice(OFFSETS)))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def classify(tts: float, high: float = 0.7, low: float = 0.005) -> str:
    if tts >= high:
        return "true-thinking"
    if tts <= low:
        return "decorative"
    return "intermediate"


# ---- scoring ---------------------------------------------------------------
@torch.no_grad()
def _answer_prob(lm: LM, prefix: str, answer: str) -> float:
    """P(answer string | prefix) = exp(Σ logp of answer tokens)."""
    pid = lm.encode(prefix)
    aid = lm.tokenizer(answer, add_special_tokens=False, return_tensors="pt").input_ids.to(lm.device)
    ids = torch.cat([pid, aid], dim=1)
    lp = lm.token_logprobs(ids)
    return float(math.exp(lp[pid.shape[1]:].sum().clamp(min=-50).cpu()))


def _prefix(problem_prompt: str, ctx: str, step: str) -> str:
    body = (ctx + ("\n" if ctx and not ctx.endswith("\n") else "") + step).strip()
    return problem_prompt + THINK_OPEN + body + THINK_CLOSE + ANSWER_TEMPLATE


def true_thinking_score(
    lm: LM,
    problem_prompt: str,
    reasoning: str,
    answer: str,
    seed: int = 42,
    high: float = 0.7,
    low: float = 0.005,
    verbose: bool = False,
) -> TTSResult:
    """Compute the True Thinking Score for each step of ``reasoning``.

    problem_prompt: the chat-formatted prompt up to (but not including) <think>.
    reasoning: the model's chain of thought (without the <think> tags).
    answer: the correct final answer string (scored inside \\boxed{...}).
    """
    rng = random.Random(seed)
    steps = segment_steps(reasoning)
    perturbed = [perturb_step(s, rng) for s in steps]      # None => removed
    ans = answer.strip() + "}"

    out = []
    for i, step in enumerate(steps):
        ctx_intact = "\n".join(steps[:i])
        ctx_pert = "\n".join(p if p is not None else "" for p in perturbed[:i])
        step_pert = perturbed[i] if perturbed[i] is not None else ""

        S11 = _answer_prob(lm, _prefix(problem_prompt, ctx_intact, step), ans)
        S10 = _answer_prob(lm, _prefix(problem_prompt, ctx_intact, step_pert), ans)
        S01 = _answer_prob(lm, _prefix(problem_prompt, ctx_pert, step), ans)
        S00 = _answer_prob(lm, _prefix(problem_prompt, ctx_pert, step_pert), ans)

        nec = abs(S11 - S10)         # intact context, perturb the step (necessity)
        suf = abs(S01 - S00)         # perturbed context, perturb the step (sufficiency)
        tts = 0.5 * (nec + suf)
        out.append(TTSStep(
            index=i, text=step, necessity=nec, sufficiency=suf, tts=tts,
            scores={"S11": S11, "S10": S10, "S01": S01, "S00": S00},
            label=classify(tts, high, low),
            self_verification=bool(_SELF_VERIFY.search(step)),
        ))
        if verbose:
            print(f"  step {i}: TTS={tts:.3f} ({out[-1].label})", flush=True)
    return TTSResult(problem=problem_prompt, answer=answer, steps=out,
                     model_name=getattr(lm.model.config, "_name_or_path", ""))


# ---- generation convenience ------------------------------------------------
@torch.no_grad()
def generate_reasoning(lm: LM, question: str, max_new_tokens: int = 1024) -> tuple[str, str]:
    """Generate a CoT for ``question`` with a thinking model. Returns
    (problem_prompt, reasoning) where problem_prompt is the chat prompt up to
    <think> and reasoning is the text inside the think block."""
    msgs = [{"role": "user", "content": question}]
    try:
        text = lm.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=True)
    except TypeError:
        text = lm.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = lm.tokenizer(text, return_tensors="pt").input_ids.to(lm.device)
    gen = lm.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=lm.baseline_token_id)
    cont = lm.tokenizer.decode(gen[0, ids.shape[1]:])
    reasoning = cont.split("</think>")[0].replace("<think>", "").strip()
    # strip a trailing <think> the chat template may have added, so the scorer
    # (which re-adds THINK_OPEN) doesn't double it
    prompt = text.rstrip()
    if prompt.endswith("<think>"):
        prompt = prompt[: -len("<think>")].rstrip() + "\n"
    return prompt, reasoning


# ---- render ----------------------------------------------------------------
def tts_chart(result: TTSResult, html_file: str | None = None):
    """Horizontal bar chart of TTS per step; true-thinking steps highlighted."""
    import html as _html

    steps = result.steps
    n = len(steps)
    W, lab, x0, row = 680, 230, 238, 26
    H = 60 + n * row
    bw = W - x0 - 70
    color = {"true-thinking": "#3a8a4a", "decorative": "#bbb", "intermediate": "#6b9bd1"}
    p = [f'<svg width="100%" viewBox="0 0 {W} {H}" role="img" font-family="ui-monospace,Menlo,monospace">',
         '<title>True Thinking Score</title><desc>causal contribution of each reasoning step</desc>',
         '<text x="20" y="22" font-size="13" fill="#222">True Thinking Score per reasoning step</text>',
         '<text x="20" y="38" font-size="10" fill="#888">green = true-thinking (≥0.7) · gray = decorative (≤0.005)</text>']
    for i, s in enumerate(steps):
        y = 52 + i * row
        w = bw * min(s.tts, 1.0)
        lbl = _html.escape((("✓ " if s.self_verification else "") + s.text)[:30])
        p.append(f'<text x="{lab}" y="{y+13}" text-anchor="end" font-size="10" fill="#444">{lbl}</text>')
        p.append(f'<rect x="{x0}" y="{y+2}" width="{max(w,1):.1f}" height="15" fill="{color[s.label]}" rx="2"><title>step {i}: {s.text[:80]}</title></rect>')
        p.append(f'<text x="{x0+max(w,1)+5:.1f}" y="{y+13}" font-size="10" fill="#888">{s.tts:.3f}</text>')
    p.append("</svg>")
    svg = "".join(p)
    if html_file:
        with open(html_file, "w") as f:
            f.write(f"<!DOCTYPE html><meta charset='utf-8'><body style='background:#fff;margin:0'>"
                    f"<div style='width:680px;max-width:100%'>{svg}</div></body>")
    try:
        from IPython.display import HTML, display
        display(HTML(svg))
    except ImportError:
        if not html_file:
            raise RuntimeError("not in IPython; pass html_file=") from None
    return svg
