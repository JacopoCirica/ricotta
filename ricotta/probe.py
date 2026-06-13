"""Residual-stream belief probing for chain-of-thought.

Reproduces the method of "Exploring belief states in LLM chains of thought":
does a model hold a linearly-decodable belief about its final answer *during*
the chain of thought, or is the answer computed only at the very end?

Pipeline: for each binary-choice question, generate a CoT, record the model's
answer, and capture residual-stream activations at chosen layers for the last N
token positions. Then, for every (layer, position), train a logistic probe to
decode the answer label with K-fold cross-validation. The resulting accuracy
grid shows *when* (token position) and *where* (layer) the belief becomes
decodable. A logit-lens of the probe direction names what it reads.

Probing decodes what is *represented*; it is the representational complement to
``attrib``'s causal CoT span-ablation — two angles on "does the model decide
late?". Needs the ``probe`` extra (scikit-learn).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Question:
    text: str
    option_a: str
    option_b: str
    truth: str | None = None        # 'A' | 'B' | None

    def prompt(self) -> str:
        return (f"Question: {self.text}\n"
                f"A) {self.option_a}\nB) {self.option_b}\n"
                f"Think step by step in a few sentences, then end with exactly "
                f"'Answer: A' or 'Answer: B'.\n")


@dataclass
class BeliefResult:
    accuracy: np.ndarray            # (n_layers, n_positions) CV probe accuracy
    layers: list[int]
    offsets: list[int]              # token offsets from end (e.g. -40..-1)
    n_questions: int
    baseline: float                 # majority-class accuracy
    label_balance: tuple[int, int]  # (#A, #B)
    logit_lens: dict = field(default_factory=dict)   # layer -> top tokens of probe dir
    model_name: str = ""

    def peak(self) -> tuple[int, int, float]:
        i, j = np.unravel_index(int(self.accuracy.argmax()), self.accuracy.shape)
        return self.layers[i], self.offsets[j], float(self.accuracy[i, j])

    def temporal(self) -> np.ndarray:
        """Best-over-layers accuracy at each position."""
        return self.accuracy.max(axis=0)

    def __repr__(self):
        l, o, a = self.peak()
        return (f"BeliefResult(n={self.n_questions}, layers={len(self.layers)}, "
                f"peak acc={a:.2f} @ layer {l} offset {o}, baseline={self.baseline:.2f})")


# ---- activation extraction -------------------------------------------------
_ANS = re.compile(r"[Aa]nswer\s*[:\-]?\s*\(?([AB])\)?")


def _parse_answer(text: str) -> str | None:
    m = list(_ANS.finditer(text))
    if m:
        return m[-1].group(1).upper()
    tail = re.findall(r"\b([AB])\b", text[-40:])
    return tail[-1] if tail else None


@torch.no_grad()
def _generate_and_extract(lm, q: Question, layers, n_positions, max_new_tokens):
    """Generate a CoT, parse the model's answer, and return residual-stream
    activations (n_layers, n_positions, d) for the last n_positions tokens."""
    ids = lm.tokenizer(q.prompt(), return_tensors="pt").input_ids.to(lm.device)
    gen = lm.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=lm.baseline_token_id)
    full = gen[0]
    cont = full[ids.shape[1]:]
    answer = _parse_answer(lm.tokenizer.decode(cont))
    if answer is None or cont.shape[0] < n_positions:
        return None, None
    out = lm.model(full.unsqueeze(0), output_hidden_states=True)
    hs = out.hidden_states                       # tuple (n_layers+1) of (1, seq, d)
    acts = np.stack([hs[l][0, -n_positions:].float().cpu().numpy() for l in layers])
    return acts, answer                          # (n_layers, n_positions, d)


def collect(lm, questions: list[Question], layers=None, n_positions=40,
            max_new_tokens=160, verbose=True):
    """Generate + extract for all questions. Returns (acts, labels, layers, offsets)."""
    n_layers = lm.model.config.num_hidden_layers
    if layers is None:    # ~6 evenly spaced transformer layers (1-indexed in hidden_states)
        layers = sorted(set(np.linspace(1, n_layers, 6).round().astype(int).tolist()))
    A, y = [], []
    for i, q in enumerate(questions):
        acts, ans = _generate_and_extract(lm, q, layers, n_positions, max_new_tokens)
        if acts is None:
            continue
        A.append(acts)
        y.append(1 if ans == "B" else 0)
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(questions)} processed, {len(A)} usable", flush=True)
    if not A:
        raise RuntimeError("no questions produced a parseable answer + enough tokens")
    offsets = list(range(-n_positions, 0))
    return np.stack(A), np.array(y), list(layers), offsets


# ---- probing ---------------------------------------------------------------
def _probe_grid(acts: np.ndarray, labels: np.ndarray, n_splits=3, seed=0):
    """acts (N, L, P, d), labels (N,) -> accuracy grid (L, P) via logistic + KFold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    N, L, P, d = acts.shape
    grid = np.full((L, P), np.nan)
    n_splits = min(n_splits, np.bincount(labels).min())
    if n_splits < 2:
        return grid
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for li in range(L):
        for pj in range(P):
            X = acts[:, li, pj, :]
            accs = []
            for tr, te in skf.split(X, labels):
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(C=0.1, max_iter=2000)
                clf.fit(sc.transform(X[tr]), labels[tr])
                accs.append(clf.score(sc.transform(X[te]), labels[te]))
            grid[li, pj] = float(np.mean(accs))
    return grid


def _logit_lens(lm, acts, labels, layers, offset_idx=-1, top_k=8):
    """Train a probe on the best position and project its direction through the
    unembedding to name what the belief direction reads."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    lens = {}
    W_U = lm.model.get_output_embeddings().weight.detach().float()   # (vocab, d)
    for li, layer in enumerate(layers):
        X = acts[:, li, offset_idx, :]
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X), labels)
        w = torch.tensor(clf.coef_[0] / sc.scale_, dtype=torch.float32, device=W_U.device)
        scores = (W_U @ w).float().cpu()
        top = scores.topk(top_k).indices.tolist()
        lens[layer] = [lm.tokenizer.decode([t]).strip() for t in top]
    return lens


def run_belief_experiment(lm, questions: list[Question], layers=None, n_positions=40,
                          max_new_tokens=160, n_splits=3, verbose=True,
                          cache_path: str | None = None) -> BeliefResult:
    """Full pipeline: generate, extract, probe per (layer, position), logit-lens.
    ``cache_path`` saves/loads the (expensive) generated activations as .npz."""
    if cache_path and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        acts, labels = d["acts"], d["labels"]
        layers, offsets = list(d["layers"]), list(d["offsets"])
    else:
        acts, labels, layers, offsets = collect(
            lm, questions, layers, n_positions, max_new_tokens, verbose)
        if cache_path:
            np.savez(cache_path, acts=acts, labels=labels, layers=layers, offsets=offsets)
    grid = _probe_grid(acts, labels, n_splits=n_splits)
    nB = int(labels.sum())
    baseline = max(nB, len(labels) - nB) / len(labels)
    lens = _logit_lens(lm, acts, labels, layers)
    return BeliefResult(accuracy=grid, layers=layers, offsets=offsets,
                        n_questions=len(labels), baseline=baseline,
                        label_balance=(len(labels) - nB, nB), logit_lens=lens,
                        model_name=getattr(lm.model.config, "_name_or_path", ""))


# ---- render ----------------------------------------------------------------
def belief_curve(result: BeliefResult, html_file: str | None = None):
    """SVG line chart: best-over-layers probe accuracy vs token offset from end."""
    acc = result.temporal()
    off = result.offsets
    W, H, ml, mt = 680, 300, 60, 30
    pw, ph = W - ml - 20, H - mt - 50
    xs = [ml + pw * i / (len(off) - 1) for i in range(len(off))]
    def y(a): return mt + ph * (1 - (a - 0.4) / 0.6)   # scale 0.4..1.0
    pts = " ".join(f"{xs[i]:.1f},{y(acc[i]):.1f}" for i in range(len(off)))
    base_y = y(result.baseline)
    parts = [f'<svg width="100%" viewBox="0 0 {W} {H}" role="img" font-family="ui-monospace,monospace">',
             f'<title>belief decodability vs position</title><desc>probe accuracy across the last token positions</desc>',
             f'<text x="{ml}" y="18" font-size="13" fill="var(--color-text-primary)">When is the answer decodable? — {result.model_name}</text>']
    for a in (0.5, 0.75, 1.0):
        parts.append(f'<line x1="{ml}" y1="{y(a):.1f}" x2="{ml+pw}" y2="{y(a):.1f}" stroke="#e3e3ea" stroke-width="1"/>')
        parts.append(f'<text x="{ml-8}" y="{y(a)+3:.1f}" text-anchor="end" font-size="10" fill="var(--color-text-secondary)">{a:.2f}</text>')
    parts.append(f'<line x1="{ml}" y1="{base_y:.1f}" x2="{ml+pw}" y2="{base_y:.1f}" stroke="#c98" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{ml+pw}" y="{base_y-4:.1f}" text-anchor="end" font-size="10" fill="#c87">baseline {result.baseline:.2f}</text>')
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#3a6ea8" stroke-width="2"/>')
    for lbl in (off[0], off[len(off)//2], off[-1]):
        i = off.index(lbl)
        parts.append(f'<text x="{xs[i]:.1f}" y="{mt+ph+16}" text-anchor="middle" font-size="10" fill="var(--color-text-secondary)">{lbl}</text>')
    parts.append(f'<text x="{ml+pw/2}" y="{H-6}" text-anchor="middle" font-size="11" fill="var(--color-text-secondary)">token offset from end of chain</text>')
    parts.append("</svg>")
    svg = "".join(parts)
    if html_file:
        with open(html_file, "w") as f:
            f.write(f"<!DOCTYPE html><meta charset='utf-8'><body style='background:#fff'>{svg}</body>")
    try:
        from IPython.display import HTML, display
        display(HTML(svg))
    except ImportError:
        pass
    return svg


# ---- a small built-in binary set (swap in MMLU-Pro for the real thing) -----
DEMO_QUESTIONS = [
    Question("Is water (H2O) a compound?", "Yes", "No", "A"),
    Question("Is the sun a planet?", "Yes", "No", "B"),
    Question("Is 17 a prime number?", "Yes", "No", "A"),
    Question("Does sound travel faster than light?", "Yes", "No", "B"),
    Question("Is a whale a mammal?", "Yes", "No", "A"),
    Question("Is iron magnetic?", "Yes", "No", "A"),
    Question("Is Mount Everest taller than K2?", "Yes", "No", "A"),
    Question("Is helium heavier than air?", "Yes", "No", "B"),
    Question("Does the Earth orbit the Sun?", "Yes", "No", "A"),
    Question("Is a tomato botanically a vegetable?", "Yes", "No", "B"),
    Question("Is 100 divisible by 7?", "Yes", "No", "B"),
    Question("Is Antarctica a desert?", "Yes", "No", "A"),
    Question("Do spiders have six legs?", "Yes", "No", "B"),
    Question("Is gold a chemical element?", "Yes", "No", "A"),
    Question("Is the Pacific the smallest ocean?", "Yes", "No", "B"),
    Question("Does ice float on water?", "Yes", "No", "A"),
]
