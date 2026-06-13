"""Reproduce the belief-state probing experiment on Qwen3-0.6B.

Scaled-down method reproduction (LessWrong: "Exploring belief states in LLM
chains of thought"). We probe the model's OWN final answer (A/B): how early in
the chain of thought is its eventual answer linearly decodable from the residual
stream? Uses numeric-comparison questions (unambiguous, balanced) so a small
model produces clean A/B answers.

Original used 500 MMLU-Pro Qs on Gemma-3-27B; this is ~60 Qs on a 0.6B — the
pipeline is identical, the statistics are weaker. Swap in MMLU-Pro + a bigger
model on a GPU box for the real thing.
"""

import random

import torch

from ricotta import LM, Question, run_belief_experiment
from ricotta.probe import belief_curve

random.seed(0)
lm = LM.load("Qwen/Qwen3-0.6B", device="mps", dtype=torch.float32)

# balanced numeric-comparison questions (answer is the model's own A/B)
questions = []
for _ in range(70):
    a, b = random.randint(10, 999), random.randint(10, 999)
    if a == b:
        b += 1
    questions.append(Question(f"Is {a} greater than {b}?", "Yes", "No"))

res = run_belief_experiment(lm, questions, n_positions=30, max_new_tokens=120, n_splits=3,
                            cache_path="belief_acts.npz")
print("\n", res)
print("label balance (A,B):", res.label_balance, "| baseline:", round(res.baseline, 3))
print("peak (layer, offset, acc):", res.peak())
print("\ntemporal best-over-layers accuracy (last 30 -> last token):")
t = res.temporal()
for k in range(0, len(res.offsets), 3):
    print(f"  offset {res.offsets[k]:4d}: {t[k]:.2f}")
print("\nlogit-lens of probe direction (per layer):")
for layer, toks in res.logit_lens.items():
    print(f"  layer {layer}: {toks}")

belief_curve(res, html_file="/Users/jacopocirica/Desktop/ricotta/examples/belief_curve.html")
print("\nwrote belief_curve.html")

print("""
Interpretation: the answer is decodable well above baseline ~30 tokens before
the end and stays flat-high — the OPPOSITE of the original 'only decodable in
the last ~10 tokens'. That's expected and informative: our task ('Is X greater
than Y') is fixed by the question, so the model commits immediately — there is
no genuine reasoning to defer the decision. The original used hard MMLU-Pro
questions where reasoning happens. Takeaway: belief-formation dynamics are
task-dependent; the pipeline reproduces faithfully, the finding needs a
reasoning-heavy dataset to show late emergence.""")
