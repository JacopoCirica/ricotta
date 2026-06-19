"""True Thinking Score demo on Qwen3.5-4B (GSM8K-style numeric problems).

Generates a reasoning chain, then scores how causally load-bearing each step is
(necessity + sufficiency under numeric perturbation), classifying steps as
true-thinking vs decorative. Reproduces the Thinking Machines TTS recipe locally.
"""
import torch

from ricotta import LM, true_thinking_score, tts_chart
from ricotta.tts import generate_reasoning

lm = LM.load("Qwen/Qwen3.5-4B", device="mps", dtype=torch.bfloat16)

PROBLEMS = [
    ("A store has 33 red, 19 blue, and 14 green marbles. It sells 6 red, 4 blue, "
     "and 2 green. How many marbles are left?", "54"),
    ("Tom has 3 boxes with 8 pencils each. He gives away 5 pencils. "
     "How many pencils does he have now?", "19"),
]

for q, ans in PROBLEMS:
    print("\n" + "=" * 70 + f"\nQ: {q}\n   correct answer: {ans}", flush=True)
    prompt, reasoning = generate_reasoning(lm, q, max_new_tokens=768)
    print("--- reasoning (first 500 chars) ---\n", reasoning[:500], "\n---", flush=True)
    res = true_thinking_score(lm, prompt, reasoning, ans, verbose=True)
    print(res)
    for s in res.steps:
        sv = " [self-verify]" if s.self_verification else ""
        print(f"  step {s.index} TTS={s.tts:.3f} ({s.label}){sv}  nec={s.necessity:.3f} suf={s.sufficiency:.3f}")
        print(f"      {s.text[:80]!r}")
    safe = "".join(c if c.isalnum() else "_" for c in q[:20])
    tts_chart(res, html_file=f"/Users/jacopocirica/Desktop/ricotta/examples/tts_{safe}.html")

print("\nwrote tts_*.html")
