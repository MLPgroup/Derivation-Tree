#!/usr/bin/env python3
"""
arxiv_counts_fixed.py

Fixes the "used prior to global declaration" error by avoiding globals in main().
Requires: requests, feedparser
pip install requests feedparser
"""
import re
import time
import json
import csv
import argparse
from urllib.parse import quote_plus

import requests
import feedparser

# --------------------------
# Config / small mappings
# --------------------------
ARXIV_API = "http://export.arxiv.org/api/query"
DEFAULT_BATCH_SIZE = 25
DEFAULT_SLEEP_BETWEEN_BATCHES = 1.2
USER_AGENT = "arxiv-hierarchy-checker/1.0 (your-email@example.com)"  # replace with your email

PREFIX_MAP = {
    "cs": "Computer Science",
    "math": "Mathematics",
    "cond-mat": "Condensed Matter",
    "hep-th": "High Energy Physics - Theory",
    "hep-ph": "High Energy Physics - Phenomenology",
    "astro-ph": "Astrophysics",
    "quant-ph": "Quantum Physics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "stat": "Statistics",
    "nlin": "Nonlinear Sciences",
    "physics": "Physics",
    "econ": "Economics",
}

SUFFIX_MAP = {
    "mes-hall": "Mesoscale and Nanoscale Physics",
    "pop-ph": "Popular Physics",
    "hep-ex": "High Energy Physics - Experiment",
    "cond-mat.mes-hall": "Mesoscale and Nanoscale Physics",
}

# --------------------------
# Helpers
# --------------------------
def normalize_id(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    s = re.sub(r"https?://arxiv\.org/(abs|pdf)/", "", s, flags=re.I)
    s = re.sub(r"^arxiv:\s*", "", s, flags=re.I)
    s = re.sub(r"v\d+$", "", s, flags=re.I)
    m = re.match(r"^(?P<prefix>[a-z\-]+)(?P<number>\d{7})$", s)
    if m:
        s = f"{m.group('prefix')}/{m.group('number')}"
    return s

def parse_entry_categories(entry):
    primary = None
    all_cats = []
    if hasattr(entry, "arxiv_primary_category") and entry.arxiv_primary_category:
        primary = entry.arxiv_primary_category.get("term")
    if hasattr(entry, "tags") and entry.tags:
        for t in entry.tags:
            if isinstance(t, dict) and "term" in t:
                all_cats.append(t["term"])
    all_cats = list(dict.fromkeys(all_cats))
    return primary, all_cats

def code_to_hierarchy(code):
    if not code:
        return None, None
    if "." in code:
        prefix, suffix = code.split(".", 1)
    elif "/" in code:
        prefix = code.split("/", 1)[0]
        suffix = None
    else:
        prefix = code
        suffix = None
    higher = PREFIX_MAP.get(prefix, prefix)
    lower = None
    if suffix:
        maybe_full = f"{prefix}.{suffix}"
        if maybe_full in SUFFIX_MAP:
            lower = SUFFIX_MAP[maybe_full]
        else:
            lower = SUFFIX_MAP.get(suffix, suffix.replace("-", " "))
    return higher, lower

# --------------------------
# Main functionality
# --------------------------
def process_ids(article_ids, batch_size=DEFAULT_BATCH_SIZE, sleep_between_batches=DEFAULT_SLEEP_BETWEEN_BATCHES):
    ids_norm = [normalize_id(x) for x in article_ids]
    results = []
    not_found = []

    for i in range(0, len(ids_norm), batch_size):
        batch = ids_norm[i:i+batch_size]
        if not batch:
            continue
        id_list_param = ",".join(batch)
        url = ARXIV_API + "?id_list=" + quote_plus(id_list_param)
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        except Exception as e:
            for orig in batch:
                results.append({"id": orig, "found": False, "error": f"network error: {e}"})
            time.sleep(sleep_between_batches)
            continue

        if resp.status_code != 200:
            for orig in batch:
                results.append({"id": orig, "found": False, "error": f"HTTP {resp.status_code}"})
            time.sleep(sleep_between_batches)
            continue

        feed = feedparser.parse(resp.text)
        entries_by_id = {}
        for e in feed.entries:
            entry_id_url = e.get("id", "")
            m = re.search(r'arxiv\.org/(abs|pdf)/(?P<aid>.+)$', entry_id_url)
            entry_id = None
            if m:
                entry_id = m.group("aid")
                entry_id = re.sub(r"v\d+$", "", entry_id)
            if entry_id:
                entries_by_id[entry_id] = e

        for orig in batch:
            rec = {"id": orig}
            entry = entries_by_id.get(orig)
            if entry is None:
                rec.update({"found": False, "error": "not found in API response"})
                not_found.append(orig)
            else:
                primary_code, all_codes = parse_entry_categories(entry)
                higher, lower = code_to_hierarchy(primary_code)
                rec.update({
                    "found": True,
                    "title": entry.get("title", "").strip(),
                    "primary_category_code": primary_code,
                    "all_category_codes": all_codes,
                    "higher_level_primary": higher,
                    "lower_level_primary": lower
                })
            results.append(rec)

        time.sleep(sleep_between_batches)

    primary_higher_counts = {}
    primary_lower_counts = {}
    all_higher_counts = {}
    all_lower_counts = {}

    for r in results:
        if not r.get("found"):
            continue
        h = r.get("higher_level_primary") or "Unknown"
        l = r.get("lower_level_primary") or "Unknown"
        primary_higher_counts[h] = primary_higher_counts.get(h, 0) + 1
        primary_lower_counts[l] = primary_lower_counts.get(l, 0) + 1

        for code in r.get("all_category_codes", []):
            hh, ll = code_to_hierarchy(code)
            hh = hh or "Unknown"
            ll = ll or "Unknown"
            all_higher_counts[hh] = all_higher_counts.get(hh, 0) + 1
            all_lower_counts[ll] = all_lower_counts.get(ll, 0) + 1

    counts = {
        "primary_higher_counts": primary_higher_counts,
        "primary_lower_counts": primary_lower_counts,
        "all_higher_counts": all_higher_counts,
        "all_lower_counts": all_lower_counts,
        "not_found_ids": not_found,
    }

    return results, counts

# --------------------------
# I/O
# --------------------------
def load_article_ids_from_file(json_file):
    with open(json_file, "r", encoding="utf8") as f:
        j = json.load(f)
    articles = j.get("Manually Parsed Articles", [])
    ids = [a.get("Article ID") for a in articles if a.get("Article ID")]
    return ids

def write_outputs(results, counts, out_prefix):
    with open(out_prefix + "_results.json", "w", encoding="utf8") as f:
        json.dump({"results": results, "counts": counts}, f, indent=2, ensure_ascii=False)

    csvfile = out_prefix + "_results.csv"
    with open(csvfile, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "found", "title", "primary_category_code",
                         "all_category_codes", "higher_level_primary", "lower_level_primary", "error"])
        for r in results:
            writer.writerow([
                r.get("id"),
                r.get("found", False),
                r.get("title", ""),
                r.get("primary_category_code", "") or "",
                ";".join(r.get("all_category_codes", [])) if r.get("all_category_codes") else "",
                r.get("higher_level_primary", "") or "",
                r.get("lower_level_primary", "") or "",
                r.get("error", "") or ""
            ])

    with open(out_prefix + "_counts.json", "w", encoding="utf8") as f:
        json.dump(counts, f, indent=2, ensure_ascii=False)

    print(f"Wrote: {out_prefix}_results.json, {csvfile}, {out_prefix}_counts.json")

def main():
    parser = argparse.ArgumentParser(description="Query arXiv for subject hierarchy counts.")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file (your format)")
    parser.add_argument("--out", "-o", default="arxiv_out", help="Output file prefix")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for API id_list queries")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_BETWEEN_BATCHES, help="Seconds between batches")
    args = parser.parse_args()

    article_ids = load_article_ids_from_file(args.input)
    if not article_ids:
        print("No Article ID entries found in JSON under 'Manually Parsed Articles'. Exiting.")
        return

    batch_size = args.batch
    sleep_between_batches = args.sleep

    print(f"Found {len(article_ids)} Article IDs. Processing in batches of {batch_size}...")

    results, counts = process_ids(article_ids, batch_size=batch_size, sleep_between_batches=sleep_between_batches)
    write_outputs(results, counts, args.out)

    # print quick summary
    print("\nPrimary higher-level counts (top-level subjects):")
    for k, v in sorted(counts["primary_higher_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    print("\nPrimary lower-level counts (subfields):")
    for k, v in sorted(counts["primary_lower_counts"].items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}")
    if counts["not_found_ids"]:
        print(f"\n{len(counts['not_found_ids'])} IDs not found. Example failures: {counts['not_found_ids'][:10]}")

if __name__ == "__main__":
    main()
