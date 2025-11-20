import os
import json

import article_parser
import derivation_graph

GEMPATH = './outputs/Gemini/combine_2025-09-28_19-41-38_UTC.json'
GEMOUT = './outputs/Gemini/postprocess/'
# CHATPATH = './outputs/Chatgpt/combine/combine_chatgpt_2025-10-03_05-25-56_UTC.json'
CHATPATH = './outputs/Chatgpt/chatgpt_2025-10-03_02-49-19_UTC.json'
CHATOUT = './outputs/Chatgpt/postprocess/'

TP_TIMEZONE = 'UTC'


def remove_redundant_edges(adj):
    """
    Remove transitive/redundant edges from directed adjacency dict.
    adj: {node: [neighbors]}
    Returns new_adj where edge u->v is removed if there exists an alternative path u -> ... -> v (length >= 2).
    """
    new_adj = {u: list(vs) for u, vs in adj.items()}
    nodes = set(new_adj.keys()) | {t for vs in new_adj.values() for t in vs}

    for u in list(nodes):
        neighbors = list(new_adj.get(u, []))
        for v in neighbors:
            # Temporarily ignore direct edge u->v and check reachability u -> ... -> v
            stack = [n for n in new_adj.get(u, []) if n != v]
            seen = set(stack)
            reachable = False
            while stack:
                cur = stack.pop()
                if cur == v:
                    reachable = True
                    break
                for nxt in new_adj.get(cur, []):
                    if nxt not in seen and nxt != u:
                        seen.add(nxt)
                        stack.append(nxt)
            if reachable:
                if v in new_adj.get(u, []):
                    new_adj[u].remove(v)
    for n in nodes:
        new_adj.setdefault(n, [])
    return new_adj


def process_file(inpath, outdir, algo_type_label):
    # Load output JSON
    with open(inpath, 'r') as f:
        data = json.load(f)

    results = data.get("Results", {})
    if not results:
        print(f"No 'Results' found in {inpath}")
        return

    article_ids = []
    predicted_adj_lists = []
    for art_key, art_val in results.items():
        if art_key.startswith("Article ID:"):
            art_id = art_key.split("Article ID:")[1].strip()
        else:
            art_id = art_key
        article_ids.append(art_id)
        pred = art_val.get("Adjacency List", {})
        predicted_adj_lists.append(pred)

    post_predicted = []
    for adj in predicted_adj_lists:
        cleaned = {str(k): list(v) if isinstance(v, list) else (list(v) if v is not None else []) for k, v in adj.items()}
        reduced = remove_redundant_edges(cleaned)
        post_predicted.append(reduced)

    articles = article_parser.get_manually_parsed_articles()
    true_adj_lists = []
    for art_id in article_ids:
        if art_id in articles:
            true_adj_lists.append(articles[art_id]["Adjacency List"])
        else:
            true_adj_lists.append({})

    (accuracies, precisions, recalls, f1_scores,
     overall_accuracy, overall_precision, overall_recall, overall_f1_score, num_skipped) = derivation_graph.evaluate_adjacency_lists(true_adj_lists, post_predicted)

    # Use same basename as input (no date)
    out_name = os.path.basename(inpath)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, out_name)

    overall_correctness = {
        "Number of articles used": len(true_adj_lists) - num_skipped,
        "Overall Correctness": {
            "Overall Accuracy": overall_accuracy,
            "Overall Precision": overall_precision,
            "Overall Recall": overall_recall,
            "Overall F1 Score": overall_f1_score
        }
    }

    article_data = {}
    for art_id, pred_adj, a, p, r, f in zip(article_ids, post_predicted, accuracies, precisions, recalls, f1_scores):
        article_data[f"Article ID: {art_id}"] = {
            "Adjacency List": pred_adj,
            "Accuracy": a,
            "Precision": p,
            "Recall": r,
            "F1 Score": f
        }

    out_json = {
        "Correctness": overall_correctness,
        "Results": article_data
    }

    with open(outpath, 'w') as f:
        json.dump(out_json, f, indent=4)

    print(f"Wrote postprocessed results to {outpath}")


def remRedundant():
    if os.path.exists(GEMPATH):
        process_file(GEMPATH, GEMOUT, "gemini")
    else:
        print(f"{GEMPATH} not found")

    if os.path.exists(CHATPATH):
        process_file(CHATPATH, CHATOUT, "chatgpt")
    else:
        print(f"{CHATPATH} not found")


if __name__ == "__main__":
    remRedundant()