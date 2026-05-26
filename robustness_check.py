"""
Prompt-selection robustness check.

Computes pooled F1 for stage-2 LLM zero-shot results on:
  - full 63-article set
  - 20-article prompt-selection subset
  - 43-article complement

Reports absolute F1 values and the gap between complement and full-set.
"""

import json
import os
from pathlib import Path

BASE = Path(__file__).parent
RESULTS = BASE / "results"
STAGE2 = RESULTS / "stage2_final"

with open(RESULTS / "prompt_selection_articles.json") as f:
    SELECTION_SET = set(json.load(f)["selection_subset"])

MODELS = {
    "gpt-5-mini":                          "gpt-5-mini",
    "gemini-3-flash-preview":              "gemini-3-flash-preview",
    "deepseek-ai_DeepSeek-V4-Flash":       "deepseek-ai/DeepSeek-V4-Flash",
    "meta-llama_Llama-3.1-8B-Instruct":   "meta-llama/Llama-3.1-8B-Instruct",
}
N_RUNS = 3


def pooled_f1(tp_sum, fp_sum, fn_sum):
    p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def load_per_article_avg(model_dir: Path, n_runs: int) -> dict[str, dict]:
    """Return {article_id: {tp, fp, fn}} averaged across runs."""
    articles: dict[str, list] = {}
    for run in range(1, n_runs + 1):
        run_dir = model_dir / f"run_{run}"
        if not run_dir.exists():
            continue
        for mf in run_dir.glob("*_metrics.json"):
            aid = mf.name.replace("_metrics.json", "")
            with open(mf) as f:
                d = json.load(f)
            if aid not in articles:
                articles[aid] = []
            articles[aid].append((d["tp"], d["fp"], d["fn"]))

    avg: dict[str, dict] = {}
    for aid, runs in articles.items():
        avg[aid] = {
            "tp": sum(r[0] for r in runs) / len(runs),
            "fp": sum(r[1] for r in runs) / len(runs),
            "fn": sum(r[2] for r in runs) / len(runs),
        }
    return avg


def compute_subset_metrics(per_article: dict, subset: set | None = None):
    """Pool tp/fp/fn over articles (or a subset) and return (P, R, F1, n_articles)."""
    items = {k: v for k, v in per_article.items() if subset is None or k in subset}
    tp = sum(v["tp"] for v in items.values())
    fp = sum(v["fp"] for v in items.values())
    fn = sum(v["fn"] for v in items.values())
    p, r, f = pooled_f1(tp, fp, fn)
    return p * 100, r * 100, f * 100, len(items)


results = {}
for dir_name, display_name in MODELS.items():
    model_dir = STAGE2 / dir_name
    if not model_dir.exists():
        continue
    per_article = load_per_article_avg(model_dir, N_RUNS)

    all_articles = set(per_article.keys())
    complement = all_articles - SELECTION_SET
    overlap = all_articles & SELECTION_SET

    full_p, full_r, full_f, full_n = compute_subset_metrics(per_article)
    comp_p, comp_r, comp_f, comp_n = compute_subset_metrics(per_article, complement)
    sel_p, sel_r, sel_f, sel_n = compute_subset_metrics(per_article, overlap)

    results[display_name] = {
        "full":       {"p": full_p, "r": full_r, "f1": full_f, "n": full_n},
        "selection":  {"p": sel_p,  "r": sel_r,  "f1": sel_f,  "n": sel_n},
        "complement": {"p": comp_p, "r": comp_r, "f1": comp_f, "n": comp_n},
        "gap_f1":     comp_f - full_f,
    }

# Print table
print("\nPrompt-Selection Robustness Check")
print("=" * 82)
hdr = f"{'Model':<40} {'Set':>4}  {'P':>6}  {'R':>6}  {'F1':>6}  {'gap':>6}"
print(hdr)
print("-" * 82)
for model, m in results.items():
    short = model.split("/")[-1]
    for label, key in [("full", "full"), ("sel.", "selection"), ("comp.", "complement")]:
        s = m[key]
        gap = f"{m['gap_f1']:+.2f}" if label == "comp." else ""
        print(f"{short if label == 'full' else '':<40} {label:>4}  {s['p']:6.2f}  {s['r']:6.2f}  {s['f1']:6.2f}  {gap:>6}")
    print()

# Save JSON for paper
out = {}
for model, m in results.items():
    out[model] = {
        "full_f1":       round(m["full"]["f1"], 2),
        "complement_f1": round(m["complement"]["f1"], 2),
        "selection_f1":  round(m["selection"]["f1"], 2),
        "gap_complement_vs_full": round(m["gap_f1"], 2),
        "n_full":        m["full"]["n"],
        "n_complement":  m["complement"]["n"],
        "n_selection":   m["selection"]["n"],
    }

out_path = RESULTS / "robustness_check.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {out_path}")
