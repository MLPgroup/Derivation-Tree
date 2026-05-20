#!/usr/bin/env python3
"""
Recalculate adjacency-list evaluation metrics for predicted graphs.

Usage:
    python recalculate_metrics.py articles.json predicted.json recalculated_results.json
"""

import json
import math
import sys
from collections import defaultdict
from statistics import mean

def normalize_adj(adj):
    """
    Convert adjacency mapping to dict of sets, filter out nulls.
    adj: dict-like mapping node -> list-of-neighbors (may include None / "null")
    Returns dict: node -> set(neighbor strings)
    """
    out = {}
    for k, v in (adj or {}).items():
        # key might be like "S0.E1" or could be other
        if v is None:
            out[k] = set()
            continue
        # If neighbors is a single scalar (rare), wrap it
        if isinstance(v, str):
            neighbors = [v]
        else:
            neighbors = list(v)
        # filter None / "null" / "" values
        clean = set()
        for n in neighbors:
            if n is None:
                continue
            # some JSONs use the literal string "null" or empty string
            if isinstance(n, str) and n.strip().lower() == "null":
                continue
            if isinstance(n, str) and n.strip() == "":
                continue
            clean.add(n)
        out[k] = clean
    return out

def evaluate_pairwise_counts(true_adj, pred_adj, allow_self=False):
    """
    Evaluate TP/FP/FN/TN by enumerating all possible directed pairs (u->v),
    excluding self-edges unless allow_self == True.
    Inputs:
        true_adj, pred_adj: dict node -> set(neighbors)
    Returns: tp, fp, fn, tn (ints)
    """
    # collect nodes appearing as sources or targets
    nodes = set(true_adj.keys()) | set(pred_adj.keys())
    # include nodes that appear only as neighbors
    for s in list(true_adj.values()) + list(pred_adj.values()):
        nodes |= set(s)
    # remove None if any
    nodes.discard(None)

    if not allow_self:
        # if nodes <=1 still evaluate (will be zero)
        pass

    tp = fp = fn = tn = 0
    nodes_list = list(nodes)
    for u in nodes_list:
        true_neighbors = true_adj.get(u, set())
        pred_neighbors = pred_adj.get(u, set())
        for v in nodes_list:
            if not allow_self and u == v:
                continue
            in_true = (v in true_neighbors)
            in_pred = (v in pred_neighbors)
            if in_true and in_pred:
                tp += 1
            elif in_true and not in_pred:
                fn += 1
            elif not in_true and in_pred:
                fp += 1
            else:
                tn += 1
    return tp, fp, fn, tn

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def compute_metrics_from_counts(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    acc = safe_div(tp + tn, total) if total != 0 else 0.0
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) != 0 else 0.0
    return acc, prec, rec, f1

def percentile(sorted_list, q):
    """
    Return q-th percentile (q in [0,1]) of sorted_list using linear interpolation.
    If list empty -> None
    """
    if not sorted_list:
        return None
    n = len(sorted_list)
    if n == 1:
        return sorted_list[0]
    k = (n - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_list[int(k)]
    d0 = sorted_list[f] * (c - k)
    d1 = sorted_list[c] * (k - f)
    return d0 + d1

def aggregate_stats(metric_values):
    """
    metric_values: list of floats
    returns dict with Mean, Lowest, Q1, Median, Q3, Highest
    """
    if not metric_values:
        return {"Mean": None, "Lowest": None, "25th Quartile (Q1)": None,
                "Median": None, "75th Quartile (Q3)": None, "Highest": None}
    sorted_vals = sorted(metric_values)
    return {
        "Mean": mean(sorted_vals),
        "Lowest": sorted_vals[0],
        "25th Quartile (Q1)": percentile(sorted_vals, 0.25),
        "Median": percentile(sorted_vals, 0.50),
        "75th Quartile (Q3)": percentile(sorted_vals, 0.75),
        "Highest": sorted_vals[-1],
    }

def find_ground_truth_article(manual_articles, article_id):
    """
    manual_articles: list of dicts under "Manually Parsed Articles"
    article_id: string like "0907.2648"
    return the article dict or None
    """
    for art in manual_articles:
        if art.get("Article ID") == article_id:
            return art
    return None

def main(articles_path, predicted_path, output_path):
    with open(articles_path, 'r', encoding='utf-8') as f:
        articles_data = json.load(f)

    with open(predicted_path, 'r', encoding='utf-8') as f:
        predicted_data = json.load(f)

    manual_articles = articles_data.get("Manually Parsed Articles", [])

    # predicted Results may be under "Results" or top-level; supports keys like "Article ID: 0907.2648"
    predicted_results = predicted_data.get("Results") or predicted_data

    recalculated = {
        "Correctness": {
            "Number of articles used": 0,
            "Overall Correctness": {},
            "Aggregate Correctness Statistics": {}
        },
        "Results": {},
        "Missing Ground Truth": [],
        "Skipped Predictions": []
    }

    per_acc = []
    per_prec = []
    per_rec = []
    per_f1 = []

    overall_tp = overall_fp = overall_fn = overall_tn = 0
    used_count = 0

    for pred_key, pred_entry in predicted_results.items():
        # Attempt to extract article id
        # Keys in predicted sample look like "Article ID: 0907.2648"
        if pred_key.startswith("Article ID:"):
            article_id = pred_key.split("Article ID:")[-1].strip()
        else:
            # could be directly the id or nested; check if pred_entry contains a field "Article ID"
            article_id = pred_entry.get("Article ID") if isinstance(pred_entry, dict) and "Article ID" in pred_entry else pred_key

        # locate ground truth
        gt_article = find_ground_truth_article(manual_articles, article_id)
        if gt_article is None:
            recalculated["Missing Ground Truth"].append(article_id)
            continue

        # predicted adjacency list may be stored under "Adjacency List" in predicted
        pred_adj_raw = None
        if isinstance(pred_entry, dict) and "Adjacency List" in pred_entry:
            pred_adj_raw = pred_entry["Adjacency List"]
        elif isinstance(pred_entry, dict) and "AdjacencyList" in pred_entry:
            pred_adj_raw = pred_entry["AdjacencyList"]
        else:
            # Perhaps predicted_results was simply a mapping of article_id -> adjacency list
            if isinstance(pred_entry, dict):
                # If the dict looks like adjacency mapping (keys like S0.E1)
                pred_adj_raw = pred_entry.get("Adjacency List") or pred_entry
            else:
                pred_adj_raw = pred_entry

        gt_adj_raw = gt_article.get("Adjacency List", {})

        # Normalize to dict->set and handle [null] or None entries
        gt_adj = normalize_adj(gt_adj_raw)
        pred_adj = normalize_adj(pred_adj_raw)

        tp, fp, fn, tn = evaluate_pairwise_counts(gt_adj, pred_adj, allow_self=False)
        acc, prec, rec, f1 = compute_metrics_from_counts(tp, fp, fn, tn)

        # store updated per-article result; preserve the predicted adjacency list for reference
        out_key = f"Article ID: {article_id}"
        recalculated["Results"][out_key] = {
            "Adjacency List (Predicted)": pred_adj_raw,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        }

        per_acc.append(acc)
        per_prec.append(prec)
        per_rec.append(rec)
        per_f1.append(f1)

        overall_tp += tp
        overall_fp += fp
        overall_fn += fn
        overall_tn += tn

        used_count += 1

    # overall metrics across all evaluated articles
    overall_total = overall_tp + overall_fp + overall_fn + overall_tn
    overall_accuracy = safe_div(overall_tp + overall_tn, overall_total) if overall_total != 0 else 0.0
    overall_precision = safe_div(overall_tp, overall_tp + overall_fp)
    overall_recall = safe_div(overall_tp, overall_tp + overall_fn)
    overall_f1 = safe_div(2 * overall_precision * overall_recall, (overall_precision + overall_recall)) if (overall_precision + overall_recall) != 0 else 0.0

    recalculated["Correctness"]["Number of articles used"] = used_count
    recalculated["Correctness"]["Overall Correctness"] = {
        "Overall Accuracy": overall_accuracy,
        "Overall Precision": overall_precision,
        "Overall Recall": overall_recall,
        "Overall F1 Score": overall_f1
    }

    recalculated["Correctness"]["Aggregate Correctness Statistics"] = {
        "Accuracy": aggregate_stats(per_acc),
        "Precision": aggregate_stats(per_prec),
        "Recall": aggregate_stats(per_rec),
        "F1 Score": aggregate_stats(per_f1)
    }

    # Save results
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(recalculated, out_f, indent=4)

    print("Recalculation complete.")
    print(f"Articles evaluated: {used_count}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python recalculate_metrics.py articles.json predicted.json recalculated_results.json")
        sys.exit(1)
    articles_path = sys.argv[1]
    predicted_path = sys.argv[2]
    output_path = sys.argv[3]
    main(articles_path, predicted_path, output_path)
