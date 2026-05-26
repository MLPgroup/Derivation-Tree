"""
Paired bootstrap significance testing over per-article F1 scores.

Produces:
  - results/per_article_f1.csv   per-article metrics for all methods
  - results/significance_table.json   full bootstrap results
  - printed table matching the paper's §6.4 format

Bootstrap protocol (as specified in the paper):
  - 10,000 resamplings of 63 articles with replacement
  - Pooled F1 per bootstrap sample (sum tp/fp/fn → P, R, F1)
  - Two-sided p-value: fraction of samples where sign(Δ) disagrees with observed mean × 2
  - Bonferroni correction within each comparison family

For LLMs (3 runs): per-article tp/fp/fn are averaged across runs before bootstrapping.
For baselines: tp/fp/fn reconstructed from ground-truth edges in mdgd.json.
"""

import csv
import json
import os
import random
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
RESULTS = BASE / "results"
OUTPUTS = BASE / "outputs"
STAGE2 = RESULTS / "stage2_final"
STAGE3 = RESULTS / "stage3_fixes"

# ── ground truth ──────────────────────────────────────────────────────────────

def load_ground_truth() -> dict[str, set[tuple[str, str]]]:
    """Return {article_id: set of (parent, child) edge tuples}."""
    with open(BASE / "mdgd.json") as f:
        entries = json.load(f)
    gt: dict[str, set] = {}
    for entry in entries:
        aid = entry["Article ID"]
        edges: set[tuple[str, str]] = set()
        for node, children in entry["Adjacency List"].items():
            for child in children:
                if child is not None:
                    edges.add((node, child))
        gt[aid] = edges
    return gt


def adj_to_edge_set(adj: dict) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for node, children in adj.items():
        for child in (children or []):
            if child is not None:
                edges.add((node, child))
    return edges


def edges_to_tpfpfn(pred: set, truth: set) -> tuple[int, int, int]:
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    return tp, fp, fn


# ── LLM loader ────────────────────────────────────────────────────────────────

def load_llm_per_article(model_dir: Path, n_runs: int = 3) -> dict[str, dict]:
    """Average tp/fp/fn across runs for each article."""
    articles: dict[str, list] = {}
    for run in range(1, n_runs + 1):
        run_dir = model_dir / f"run_{run}"
        if not run_dir.exists():
            continue
        for mf in run_dir.glob("*_metrics.json"):
            aid = mf.name.replace("_metrics.json", "")
            with open(mf) as f:
                d = json.load(f)
            articles.setdefault(aid, []).append((d["tp"], d["fp"], d["fn"]))

    return {
        aid: {
            "tp": sum(r[0] for r in runs) / len(runs),
            "fp": sum(r[1] for r in runs) / len(runs),
            "fn": sum(r[2] for r in runs) / len(runs),
        }
        for aid, runs in articles.items()
    }


# ── baseline loaders ──────────────────────────────────────────────────────────

def load_brute_force(gt: dict[str, set]) -> dict[str, dict]:
    path = OUTPUTS / "Brute_Force" / "brute_force.json"
    with open(path) as f:
        data = json.load(f)
    result = {}
    for key, val in data["Results"].items():
        aid = key.replace("Article ID: ", "")
        pred = adj_to_edge_set(val["Adjacency List"])
        tp, fp, fn = edges_to_tpfpfn(pred, gt.get(aid, set()))
        result[aid] = {"tp": tp, "fp": fp, "fn": fn}
    return result


def load_token_similarity(gt: dict[str, set]) -> dict[str, dict]:
    # Use the primary threshold file (90%)
    path = OUTPUTS / "Token_Similarity" / "token_similarity_1_90_greater.json"
    with open(path) as f:
        data = json.load(f)
    result = {}
    for key, val in data["Results"].items():
        aid = key.replace("Article ID: ", "")
        pred = adj_to_edge_set(val["Adjacency List"])
        tp, fp, fn = edges_to_tpfpfn(pred, gt.get(aid, set()))
        result[aid] = {"tp": tp, "fp": fp, "fn": fn}
    return result


def load_naive_bayes(gt: dict[str, set]) -> dict[str, dict]:
    # Use the latest 5-fold file
    nb_dir = OUTPUTS / "Naive_Bayes"
    candidates = sorted(nb_dir.glob("naive_bayes_5fold_*.json"))
    path = candidates[-1]
    with open(path) as f:
        data = json.load(f)
    result = {}
    for key, val in data["Results"].items():
        aid = key.replace("Article ID: ", "")
        pred = adj_to_edge_set(val["Adjacency List"])
        tp, fp, fn = edges_to_tpfpfn(pred, gt.get(aid, set()))
        result[aid] = {"tp": tp, "fp": fp, "fn": fn}
    return result


# ── collect all methods ───────────────────────────────────────────────────────

def collect_all_methods(gt: dict) -> dict[str, dict[str, dict]]:
    """Return {method_label: {article_id: {tp, fp, fn}}}."""
    methods: dict[str, dict] = {}

    # Baselines
    methods["Brute Force"]     = load_brute_force(gt)
    methods["Token Similarity"] = load_token_similarity(gt)
    methods["Naive Bayes"]     = load_naive_bayes(gt)

    # Stage-2 zero-shot LLMs
    stage2_models = {
        "gpt-5-mini":                        "GPT-5-mini ZS",
        "gemini-3-flash-preview":            "Gemini ZS",
        "deepseek-ai_DeepSeek-V4-Flash":     "DeepSeek ZS",
        "meta-llama_Llama-3.1-8B-Instruct": "Llama ZS",
    }
    for dir_name, label in stage2_models.items():
        model_dir = STAGE2 / dir_name
        if model_dir.exists():
            methods[label] = load_llm_per_article(model_dir)

    # Stage-3 fixes (top-3 models only)
    stage3_models = {
        "gpt-5-mini":                    "GPT-5-mini",
        "gemini-3-flash-preview":        "Gemini",
        "deepseek-ai_DeepSeek-V4-Flash": "DeepSeek",
    }
    fixes = {
        "combination":           "Comb.",
        "edge_limit":            "Edge Limit",
        "fewshot":               "2-Shot",
        "postprocess":           "Postprocess",
        "postprocess_combination": "Comb.+Post",
    }
    for fix_dir, fix_label in fixes.items():
        fix_path = STAGE3 / fix_dir
        if not fix_path.exists():
            continue
        for dir_name, model_label in stage3_models.items():
            model_dir = fix_path / dir_name
            if model_dir.exists():
                label = f"{fix_label} - {model_label}"
                methods[label] = load_llm_per_article(model_dir)

    return methods


# ── bootstrap ─────────────────────────────────────────────────────────────────

def pooled_f1_from_articles(article_ids: list, data: dict) -> float:
    tp = sum(data[a]["tp"] for a in article_ids if a in data)
    fp = sum(data[a]["fp"] for a in article_ids if a in data)
    fn = sum(data[a]["fn"] for a in article_ids if a in data)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def per_article_f1(data: dict) -> dict:
    """Compute per-article F1 from tp/fp/fn."""
    result = {}
    for aid, m in data.items():
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        result[aid] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if (p + r) > 0 else 0.0,
        }
    return result


def bootstrap_test(
    articles: list[str],
    data_a: dict,
    data_b: dict,
    n_boot: int = 10_000,
    rng: np.random.Generator = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng(42)

    obs_a = pooled_f1_from_articles(articles, data_a)
    obs_b = pooled_f1_from_articles(articles, data_b)
    obs_delta = obs_a - obs_b

    deltas = np.empty(n_boot)
    indices = np.arange(len(articles))
    for i in range(n_boot):
        sample_idx = rng.choice(indices, size=len(articles), replace=True)
        sample = [articles[j] for j in sample_idx]
        deltas[i] = pooled_f1_from_articles(sample, data_a) - pooled_f1_from_articles(sample, data_b)

    # Two-sided p-value: fraction of bootstrap deltas that disagree with observed sign
    if obs_delta >= 0:
        p_val = 2 * np.mean(deltas < 0)
    else:
        p_val = 2 * np.mean(deltas >= 0)
    p_val = min(p_val, 1.0)

    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])

    return {
        "obs_f1_a":   obs_a * 100,
        "obs_f1_b":   obs_b * 100,
        "delta_f1":   obs_delta * 100,
        "mean_delta": np.mean(deltas) * 100,
        "ci_lo":      ci_lo * 100,
        "ci_hi":      ci_hi * 100,
        "p_value":    float(p_val),
    }


# ── comparison families ───────────────────────────────────────────────────────

COMPARISONS = [
    # (label, method_A, method_B, family)
    # Family 1: LLMs vs baselines
    ("Gemini ZS vs Brute Force",       "Gemini ZS",      "Brute Force",       "llm_vs_baseline"),
    ("DeepSeek ZS vs Brute Force",     "DeepSeek ZS",    "Brute Force",       "llm_vs_baseline"),
    ("GPT-5-mini ZS vs Brute Force",   "GPT-5-mini ZS",  "Brute Force",       "llm_vs_baseline"),
    ("Gemini ZS vs Token Similarity",  "Gemini ZS",      "Token Similarity",  "llm_vs_baseline"),
    ("DeepSeek ZS vs Token Similarity","DeepSeek ZS",    "Token Similarity",  "llm_vs_baseline"),
    ("GPT-5-mini ZS vs Token Sim.",    "GPT-5-mini ZS",  "Token Similarity",  "llm_vs_baseline"),
    ("Gemini ZS vs Naive Bayes",       "Gemini ZS",      "Naive Bayes",       "llm_vs_baseline"),
    ("DeepSeek ZS vs Naive Bayes",     "DeepSeek ZS",    "Naive Bayes",       "llm_vs_baseline"),
    ("GPT-5-mini ZS vs Naive Bayes",   "GPT-5-mini ZS",  "Naive Bayes",       "llm_vs_baseline"),
    # Family 2: best LLM vs second-best
    ("Gemini ZS vs DeepSeek ZS",       "Gemini ZS",      "DeepSeek ZS",       "llm_vs_llm"),
    ("Gemini ZS vs GPT-5-mini ZS",     "Gemini ZS",      "GPT-5-mini ZS",     "llm_vs_llm"),
    ("DeepSeek ZS vs GPT-5-mini ZS",   "DeepSeek ZS",    "GPT-5-mini ZS",     "llm_vs_llm"),
    # Family 3: fixes vs zero-shot (same model)
    ("2-Shot Gemini vs Gemini ZS",         "2-Shot - Gemini",         "Gemini ZS",     "fix_vs_zs"),
    ("2-Shot DeepSeek vs DeepSeek ZS",     "2-Shot - DeepSeek",       "DeepSeek ZS",   "fix_vs_zs"),
    ("2-Shot GPT-5-mini vs GPT-5-mini ZS", "2-Shot - GPT-5-mini",     "GPT-5-mini ZS", "fix_vs_zs"),
    ("Comb. Gemini vs Gemini ZS",              "Comb. - Gemini",          "Gemini ZS",     "fix_vs_zs"),
    ("Comb. DeepSeek vs DeepSeek ZS",          "Comb. - DeepSeek",        "DeepSeek ZS",   "fix_vs_zs"),
    ("Comb. GPT-5-mini vs GPT-5-mini ZS",      "Comb. - GPT-5-mini",      "GPT-5-mini ZS", "fix_vs_zs"),
    ("Edge Limit Gemini vs Gemini ZS",          "Edge Limit - Gemini",     "Gemini ZS",     "fix_vs_zs"),
    ("Edge Limit DeepSeek vs DeepSeek ZS",      "Edge Limit - DeepSeek",   "DeepSeek ZS",   "fix_vs_zs"),
    ("Edge Limit GPT-5-mini vs GPT-5-mini ZS",  "Edge Limit - GPT-5-mini", "GPT-5-mini ZS", "fix_vs_zs"),
    ("Comb.+Post Gemini vs Gemini ZS",          "Comb.+Post - Gemini",     "Gemini ZS",     "fix_vs_zs"),
    ("Comb.+Post DeepSeek vs DeepSeek ZS",      "Comb.+Post - DeepSeek",   "DeepSeek ZS",   "fix_vs_zs"),
    ("Comb.+Post GPT-5-mini vs GPT-5-mini ZS",  "Comb.+Post - GPT-5-mini", "GPT-5-mini ZS", "fix_vs_zs"),
]


def run_all(methods: dict) -> list[dict]:
    articles = sorted({a for data in methods.values() for a in data})
    rng = np.random.default_rng(42)

    # Count comparisons per family for Bonferroni
    from collections import Counter
    family_counts = Counter(fam for _, _, _, fam in COMPARISONS)

    rows = []
    for label, a, b, family in COMPARISONS:
        if a not in methods or b not in methods:
            continue
        res = bootstrap_test(articles, methods[a], methods[b], n_boot=10_000, rng=rng)
        alpha_bonf = 0.05 / family_counts[family]
        sig = res["p_value"] < alpha_bonf
        rows.append({
            "comparison": label,
            "method_a":   a,
            "method_b":   b,
            "family":     family,
            "n_comparisons_in_family": family_counts[family],
            "alpha_bonferroni": alpha_bonf,
            **res,
            "significant": sig,
        })
    return rows


def save_csv(methods: dict, path: Path):
    rows = []
    for method, data in methods.items():
        paf = per_article_f1(data)
        for aid, m in paf.items():
            rows.append({
                "article_id": aid,
                "method":     method,
                "seed":       "avg",
                "precision":  round(m["precision"], 6),
                "recall":     round(m["recall"], 6),
                "f1":         round(m["f1"], 6),
                "tp":         m["tp"],
                "fp":         m["fp"],
                "fn":         m["fn"],
            })
    rows.sort(key=lambda r: (r["article_id"], r["method"]))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["article_id", "method", "seed", "precision", "recall", "f1", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]):
    print("\nPaired Bootstrap Significance Test (10,000 samples, Bonferroni-corrected)")
    print("=" * 110)
    print(f"{'Comparison':<45} {'ΔF1':>6}  {'95% CI':>16}  {'p-value':>8}  {'sig (α=0.05)':>12}")
    print("-" * 110)
    cur_family = None
    for r in rows:
        if r["family"] != cur_family:
            cur_family = r["family"]
            print(f"\n  [{cur_family}]")
        ci = f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]"
        p_str = f"<0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}"
        sig_str = "Yes *" if r["significant"] else "No"
        print(f"  {r['comparison']:<43} {r['delta_f1']:+6.2f}  {ci:>16}  {p_str:>8}  {sig_str:>12}")
    print()


def main():
    print("Loading ground truth...")
    gt = load_ground_truth()

    print("Loading method results...")
    methods = collect_all_methods(gt)
    print(f"  Loaded {len(methods)} methods: {list(methods.keys())}")

    csv_path = RESULTS / "per_article_f1.csv"
    print(f"Saving per-article CSV to {csv_path}...")
    save_csv(methods, csv_path)

    print("Running bootstrap tests (10,000 samples each)...")
    rows = run_all(methods)

    print_table(rows)

    out_path = RESULTS / "significance_table.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved full results to {out_path}")


if __name__ == "__main__":
    main()
