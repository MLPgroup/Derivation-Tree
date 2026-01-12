#!/usr/bin/env python3
"""
derivation_graph_analysis.py

Classify dataset by derivation-graph length (longest directed path) and evaluate run performance per-class.

Usage:
    python derivation_graph_analysis.py --ground ground_truth.json --results run_results.json --outdir outputs [--plots]

Outputs (in outdir, timestamped):
 - per_article CSV/JSON
 - bucket_stats CSV/JSON
 - summary TXT
 - per_article_numbers TXT (one line per article with numeric columns)
 - optional PNG plots (scatter/hist/bar) if --plots is set

Requires:
    pip install pandas matplotlib
"""

import argparse
import json
from pathlib import Path
import datetime
import csv
from collections import defaultdict
import statistics

import pandas as pd

# Optional plotting
try:
    import matplotlib.pyplot as pl t
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

# -----------------------
# Helpers for parsing + graph metrics
# -----------------------
def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)

def normalize_article_key(key):
    """Normalize keys like 'Article ID: 0907.2648' -> '0907.2648'"""
    if isinstance(key, str) and ":" in key:
        _, aid = key.split(":", 1)
        return aid.strip()
    return str(key).strip()

def parse_ground_truth(gt_json):
    """
    Expects a JSON structure that contains adjacency lists for articles.
    This function tries to find adjacency lists either under a top-level
    key 'Manually Parsed Articles' (list of dicts containing 'Article ID' and 'Adjacency List')
    or under 'Results' with keys like 'Article ID: 0907.2648' (matching the accuracy format).
    Returns dict: article_id -> { 'adj': {node: [targets...]}, ... }.
    """
    out = {}

    # Case 1: top-level 'Manually Parsed Articles' list (your earlier format)
    if isinstance(gt_json, dict) and "Manually Parsed Articles" in gt_json:
        for a in gt_json["Manually Parsed Articles"]:
            aid = a.get("Article ID")
            if not aid:
                continue
            adj_raw = a.get("Adjacency List", {}) or {}
            adj = normalize_adj(adj_raw)
            out[aid] = {"adj": adj}
        return out

    # Case 2: top-level 'Results' mapping like the accuracy file, where adjacency lists exist
    if isinstance(gt_json, dict) and "Results" in gt_json:
        results = gt_json["Results"]
        for k, v in results.items():
            aid = normalize_article_key(k)
            adj_raw = v.get("Adjacency List", {}) or {}
            adj = normalize_adj(adj_raw)
            out[aid] = {"adj": adj}
        return out

    # Case 3: The ground truth might be a simple mapping {id: {...}}
    # We'll try to detect adjacency lists under each value
    if isinstance(gt_json, dict):
        for k, v in gt_json.items():
            if isinstance(v, dict) and "Adjacency List" in v:
                aid = normalize_article_key(k)
                adj = normalize_adj(v.get("Adjacency List", {}) or {})
                out[aid] = {"adj": adj}
        if out:
            return out

    raise ValueError("Could not find adjacency lists in the provided ground-truth JSON. "
                     "Expected 'Manually Parsed Articles' or top-level 'Results' or mapping of id->object containing 'Adjacency List'.")

def normalize_adj(adj_raw):
    """Convert raw adjacency list to normalized dict: node -> list of targets (None for nulls)."""
    adj = {}
    if not isinstance(adj_raw, dict):
        return adj
    for node, targets in adj_raw.items():
        if targets is None:
            adj[node] = []
            continue
        if not isinstance(targets, list):
            targets = [targets]
        normalized_targets = []
        for t in targets:
            if t is None:
                normalized_targets.append(None)
            elif isinstance(t, str) and t.lower() == "null":
                normalized_targets.append(None)
            else:
                normalized_targets.append(t)
        adj[node] = normalized_targets
    return adj

def parse_results_performance(results_json):
    """
    Parse the performance JSON mapping (the run results) into a dict:
      article_id -> { 'accuracy':..., 'precision':..., 'recall':..., 'f1':... }
    Accepts keys like 'Article ID: 0907.2648' or plain ids.
    """
    out = {}
    if isinstance(results_json, dict) and "Results" in results_json:
        results_map = results_json["Results"]
    else:
        results_map = results_json

    for k, v in results_map.items():
        aid = normalize_article_key(k)
        # if v is a dict with metrics, extract; else skip
        if not isinstance(v, dict):
            continue
        out[aid] = {
            "accuracy": v.get("Accuracy"),
            "precision": v.get("Precision"),
            "recall": v.get("Recall"),
            "f1": v.get("F1 Score")
        }
    return out

def count_edges(adj):
    """Count non-null edges in the adjacency dict."""
    total = 0
    for node, targets in adj.items():
        for t in targets:
            if t is None:
                continue
            total += 1
    return total

def longest_path_length(adj):
    """
    Compute longest directed path (in edges) in the adjacency DAG.
    Handles nodes referenced as targets but not appearing as keys.
    If cycles exist, we avoid infinite recursion by treating cycle continuation as zero-length.
    """
    # Build adjacency with None targets removed
    graph = {n: [t for t in targets if t is not None] for n, targets in adj.items()}
    # add target nodes as leaves if missing
    for targets in list(graph.values()):
        for t in targets:
            if t not in graph:
                graph[t] = []

    memo = {}
    visiting = set()

    def dfs(u):
        if u in memo:
            return memo[u]
        if u in visiting:
            # cycle detected, stop this branch
            return 0
        visiting.add(u)
        best = 0
        for v in graph.get(u, []):
            candidate = 1 + dfs(v)
            if candidate > best:
                best = candidate
        visiting.remove(u)
        memo[u] = best
        return best

    best_overall = 0
    for node in list(graph.keys()):
        val = dfs(node)
        if val > best_overall:
            best_overall = val
    return best_overall

# Default buckets for longest-path (edges)
DEFAULT_BUCKET_FN = lambda L: (
    "0-1" if L <= 1 else
    ("2" if L == 2 else
     ("3" if L == 3 else
      ("4" if L == 4 else
       ("5-6" if 5 <= L <= 6 else
        ("7-10" if 7 <= L <= 10 else
         ("11-20" if 11 <= L <= 20 else "21+")))))))

# -----------------------
# Core analysis
# -----------------------
def analyze(ground_truth_path, results_path, outdir=".", bucket_fn=DEFAULT_BUCKET_FN, plots=False):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    prefix = outdir / f"deriv_analysis_{now}"

    # Load files
    gt_json = load_json(ground_truth_path)
    res_json = load_json(results_path)

    gt_parsed = parse_ground_truth(gt_json)  # article_id -> {'adj': {...}}
    perf_parsed = parse_results_performance(res_json)  # article_id -> metrics dict

    # Build per-article stats from ground truth
    rows = []
    for aid, meta in gt_parsed.items():
        adj = meta.get("adj", {}) or {}
        n_equations = len(adj)
        n_edges = count_edges(adj)
        L = longest_path_length(adj)
        rows.append({
            "article_id": aid,
            "n_equations": n_equations,
            "n_edges": n_edges,
            "longest_path": L
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No articles parsed from ground-truth JSON.")

    # Merge performance metrics (may be missing for some ids)
    perf_df = pd.DataFrame.from_dict(perf_parsed, orient="index").reset_index().rename(columns={"index": "article_id"})
    merged = pd.merge(df, perf_df, on="article_id", how="left")

    # Bucket by longest_path
    merged["bucket"] = merged["longest_path"].apply(bucket_fn)

    # Compute bucket-level aggregates
    agg = merged.groupby("bucket").agg(
        count=("article_id", "count"),
        mean_accuracy=("accuracy", "mean"),
        median_accuracy=("accuracy", "median"),
        mean_precision=("precision", "mean"),
        median_precision=("precision", "median"),
        mean_recall=("recall", "mean"),
        median_recall=("recall", "median"),
        mean_f1=("f1", "mean"),
        median_f1=("f1", "median"),
        mean_n_equations=("n_equations", "mean"),
        mean_n_edges=("n_edges", "mean")
    ).reset_index().sort_values("count", ascending=False)

    # Save outputs
    per_csv = str(prefix) + "_per_article.csv"
    per_json = str(prefix) + "_per_article.json"
    agg_csv = str(prefix) + "_bucket_stats.csv"
    agg_json = str(prefix) + "_bucket_stats.json"
    summary_txt = str(prefix) + "_summary.txt"
    numbers_txt = str(prefix) + "_per_article_numbers.txt"

    merged.to_csv(per_csv, index=False)
    merged.to_json(per_json, orient="records", indent=2)
    agg.to_csv(agg_csv, index=False)
    agg.to_json(agg_json, orient="records", indent=2)

    # summary text
    with open(summary_txt, "w", encoding="utf8") as f:
        f.write("Derivation-graph-length classification analysis\n")
        f.write(f"Generated (UTC): {now}\n\n")
        f.write(f"Ground-truth articles parsed: {len(gt_parsed)}\n")
        f.write(f"Performance entries parsed: {len(perf_parsed)}\n")
        f.write(f"Articles with performance metrics (merged): {merged['accuracy'].count()} (non-null accuracy)\n\n")
        f.write("Overall dataset distributions (derived from ground-truth graphs):\n")
        f.write(f"  Equations per article: mean={merged['n_equations'].mean():.2f}, median={merged['n_equations'].median():.0f}, min={int(merged['n_equations'].min())}, max={int(merged['n_equations'].max())}\n")
        f.write(f"  Edges per article: mean={merged['n_edges'].mean():.2f}, median={merged['n_edges'].median():.0f}, min={int(merged['n_edges'].min())}, max={int(merged['n_edges'].max())}\n")
        f.write(f"  Longest path (edges): mean={merged['longest_path'].mean():.2f}, median={merged['longest_path'].median():.0f}, min={int(merged['longest_path'].min())}, max={int(merged['longest_path'].max())}\n\n")
        f.write("Bucket summary (bucket,count,mean_accuracy,mean_precision,mean_recall,mean_f1):\n")
        for _, row in agg.iterrows():
            f.write(f"{row['bucket']},{int(row['count'])},{row['mean_accuracy'] if not pd.isna(row['mean_accuracy']) else ''},{row['mean_precision'] if not pd.isna(row['mean_precision']) else ''},{row['mean_recall'] if not pd.isna(row['mean_recall']) else ''},{row['mean_f1'] if not pd.isna(row['mean_f1']) else ''}\n")

    # per-article numbers file
    with open(numbers_txt, "w", encoding="utf8") as f:
        f.write("article_id\tn_equations\tn_edges\tlongest_path\taccuracy\tprecision\trecall\tf1\n")
        for _, r in merged.iterrows():
            f.write(f"{r['article_id']}\t{int(r['n_equations'])}\t{int(r['n_edges'])}\t{int(r['longest_path'])}\t{r['accuracy'] if not pd.isna(r['accuracy']) else ''}\t{r['precision'] if not pd.isna(r['precision']) else ''}\t{r['recall'] if not pd.isna(r['recall']) else ''}\t{r['f1'] if not pd.isna(r['f1']) else ''}\n")

    created = {
        "per_article_csv": per_csv,
        "per_article_json": per_json,
        "bucket_csv": agg_csv,
        "bucket_json": agg_json,
        "summary_txt": summary_txt,
        "numbers_txt": numbers_txt
    }

    # Optional plots
    if plots:
        if not _HAS_PLT:
            print("matplotlib not available; skipping plots.")
        else:
            # Accuracy vs longest_path (scatter)
            fig1 = str(prefix) + "_scatter_longest_vs_accuracy.png"
            plt.figure()
            plt.scatter(merged["longest_path"], merged["accuracy"])
            plt.xlabel("Longest path (edges)")
            plt.ylabel("Accuracy")
            plt.title("Accuracy vs Longest Derivation Path")
            plt.savefig(fig1)
            plt.close()
            created["scatter_longest_vs_accuracy"] = fig1

            # Accuracy vs n_edges
            fig2 = str(prefix) + "_scatter_edges_vs_accuracy.png"
            plt.figure()
            plt.scatter(merged["n_edges"], merged["accuracy"])
            plt.xlabel("Number of edges")
            plt.ylabel("Accuracy")
            plt.title("Accuracy vs Number of Edges")
            plt.savefig(fig2)
            plt.close()
            created["scatter_edges_vs_accuracy"] = fig2

            # Histogram longest_path
            fig3 = str(prefix) + "_hist_longest_path.png"
            plt.figure()
            plt.hist(merged["longest_path"].astype(int).dropna())
            plt.xlabel("Longest path (edges)")
            plt.ylabel("Count")
            plt.title("Distribution of Longest Derivation Path Lengths")
            plt.savefig(fig3)
            plt.close()
            created["hist_longest_path"] = fig3

            # Histogram edges
            fig4 = str(prefix) + "_hist_edges.png"
            plt.figure()
            plt.hist(merged["n_edges"].astype(int).dropna())
            plt.xlabel("Number of edges")
            plt.ylabel("Count")
            plt.title("Distribution of Edges per Article")
            plt.savefig(fig4)
            plt.close()
            created["hist_edges"] = fig4

            # Bucket counts bar
            fig5 = str(prefix) + "_bar_bucket_counts.png"
            plt.figure()
            plt.bar(agg["bucket"], agg["count"])
            plt.xlabel("Bucket (longest path)")
            plt.ylabel("Article count")
            plt.title("Article counts per longest-path bucket")
            plt.savefig(fig5)
            plt.close()
            created["bar_bucket_counts"] = fig5

    return created

# -----------------------
# CLI
# -----------------------
def main():
    p = argparse.ArgumentParser(description="Classify by derivation-graph length and evaluate performance per-class.")
    p.add_argument("--ground", "-g", default="articles.json", help="Ground-truth JSON file path (contains adjacency lists)")
    p.add_argument("--results", "-r", default="./outputs/gemini/combine_2025-09-28_19-41-38_UTC.json", help="Run results JSON file path (performance metrics)")
    p.add_argument("--outdir", "-o", default="./outputs/analysis", help="Output directory")
    p.add_argument("--plots", action="store_true", help="Save plots (requires matplotlib)")
    args = p.parse_args()

    created = analyze(args.ground, args.results, outdir=args.outdir, bucket_fn=DEFAULT_BUCKET_FN, plots=args.plots)
    print("Analysis complete. Created files:")
    for k, v in created.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    main()
