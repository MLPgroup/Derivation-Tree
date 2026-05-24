#!/usr/bin/env python3
"""
llm_pipeline.py — derivation graph extraction pipeline

Stages:
  0   Fix 20-article subset (run once, seed=42)
  1   Prompt selection (20 articles, all models, P0-P7, 1 run)
  2   Final evaluation (64 articles, all models, best prompt, 3 runs)

Special modes:
  --cost-estimate   Run most expensive article through all 6 models, print upper-bound table
"""

import os, json, time, random, argparse, datetime, re, sys
from pathlib import Path

import anthropic
from openai import OpenAI
from google import genai
from google.genai import types as genai_types

import article_parser


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_CONFIG = {
    # "claude-haiku-4-5-20251001": {
    #     "provider":                   "anthropic",
    #     "model_id":                   "claude-haiku-4-5-20251001",
    #     "cost_input_per_1m":          1.00,
    #     "cost_output_per_1m":         5.00,
    #     "context_limit_tokens":       200_000,
    #     "recommended_temperature":    None,
    # },
    "gpt-5-mini": {
        "provider":                 "openai",
        "model_id":                 "gpt-5-mini",
        "cost_input_per_1m":        0.25,
        "cost_output_per_1m":       2.00,
        "context_limit_tokens":     128_000,
        "recommended_temperature":  None,
    },
    # "gemini-3.5-flash": {
    #     "provider":                 "google",
    #     "model_id":                 "gemini-3.5-flash",
    #     "cost_input_per_1m":        1.50,
    #     "cost_output_per_1m":       9.00,
    #     "context_limit_tokens":     1_048_576,
    #     "recommended_temperature":  None,
    # },
    "gemini-3-flash-preview": {
        "provider":                   "google",
        "model_id":                   "gemini-3-flash-preview",
        "cost_input_per_1m":          0.50,
        "cost_output_per_1m":         3.00,
        "cost_input_batch_per_1m":    0.25,
        "cost_output_batch_per_1m":   1.50,
        "context_limit_tokens":       1_048_576,
        "use_thinking":               True,    # thinking on by default
        "thinking_level":             "medium", # set explicitly
        "recommended_temperature":    None,
    },
    "deepseek-ai/DeepSeek-V4-Flash": {
        "provider":                   "hf",
        "model_id":                   "deepseek-ai/DeepSeek-V4-Flash",
        "hf_providers":               ["deepinfra"],
        "cost_input_per_1m":          0.14,
        "cost_output_per_1m":         0.28,
        "context_limit_tokens":       1_048_576,
        "use_thinking":               False,
        "recommended_temperature":    None,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "provider":                   "hf",
        "model_id":                   "meta-llama/Llama-3.1-8B-Instruct",
        "hf_providers":               ["nscale"],
        "cost_input_per_1m":          0.06,
        "cost_output_per_1m":         0.06,
        "context_limit_tokens":       131_072,
        "use_thinking":               False,
        "recommended_temperature":    None,
    },
    # "Qwen/Qwen3.5-9B": {
    #     "provider":                   "hf",
    #     "model_id":                   "Qwen/Qwen3.5-9B",
    #     "hf_providers":               ["together", "ovhcloud"],
    #     "cost_input_per_1m":          0.10,
    #     "cost_output_per_1m":         0.15,
    #     "context_limit_tokens":       262_144,
    #     "use_thinking":               False,
    #     "recommended_temperature":    None,
    # },
}

ALL_MODELS = list(MODEL_CONFIG.keys())

SHORT_NAME = {
    "claude-haiku-4-5-20251001":          "claude",
    "gpt-5-mini":                         "gpt-5-mini",
    "gemini-3-flash-preview":             "gemini",
    "deepseek-ai/DeepSeek-V4-Flash":      "deepseek",
    "Qwen/Qwen3.5-9B":                    "qwen",
    "meta-llama/Llama-3.1-8B-Instruct":  "llama",
}

RESULTS_DIR = Path("results")
STAGE1_DIR  = RESULTS_DIR / "stage1_prompt_selection"
STAGE2_DIR  = RESULTS_DIR / "stage2_final"

VARIANTS = [f"P{i}" for i in range(8)]

# Article whose equation set doesn't match the HTML — skip everywhere
BROKEN_ARTICLES = {"1701.00003"}

# ---------------------------------------------------------------------------
# JSON schemas for structured output
# ---------------------------------------------------------------------------

DERIVATION_GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "derivation_graph": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string"},
            },
        }
    },
    "required": ["derivation_graph"],
    "additionalProperties": False,
}

_COT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "derivation_graph": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    },
    "required": ["reasoning", "derivation_graph"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Token counting (tiktoken if available, else char/4)
# ---------------------------------------------------------------------------

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

# ---------------------------------------------------------------------------
# Article input building
# ---------------------------------------------------------------------------

def build_article_inputs(article_id, article_data, html_content):
    """
    Returns:
        ok               bool
        total_text       str   full article text with equations inline
        eq_list          str   numbered list "1. <alttext>\n2. ..."
        condensed        str   2-sentence windows around each equation (P7)
        eq_indexing      list  ordered equation IDs
        num_to_id        dict  {"1": "S2.E1", ...}
        id_to_num        dict  {"S2.E1": "1", ...}
    """
    equations, words_between, eq_indexing = article_parser.extract_equations(html_content)

    expected_ids = set(article_data["Equation ID"])
    eq_indexing = [e for e in eq_indexing if e in expected_ids]
    equations = {k: v for k, v in equations.items() if k in expected_ids}
    words_between = words_between[: len(eq_indexing) + 1]

    if set(eq_indexing) != expected_ids:
        return False, None, None, None, None, None, None

    # Equation number mapping
    eq_num_map = article_data.get("Equation Number", {})
    id_to_num = {eid: eq_num_map.get(eid, str(i + 1)) for i, eid in enumerate(eq_indexing)}
    num_to_id = {v: k for k, v in id_to_num.items()}

    # total article text (equations inline as alttext)
    total_text = words_between[0] if words_between else ""
    eq_alttexts = []
    for i, eid in enumerate(eq_indexing):
        alttext = " ".join(s["alttext"] for s in equations[eid]["equations"])
        eq_alttexts.append(alttext)
        total_text += " " + alttext
        if i + 1 < len(words_between):
            total_text += " " + words_between[i + 1]

    # equation list string
    eq_list = "\n".join(f"{id_to_num[eid]}. {eq_alttexts[i]}" for i, eid in enumerate(eq_indexing))

    # condensed context (P7): 2 sentences before + 2 sentences after each equation
    def last_n_sents(text, n):
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return " ".join(parts[-n:]) if parts else ""

    def first_n_sents(text, n):
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return " ".join(parts[:n]) if parts else ""

    condensed_parts = []
    for i, eid in enumerate(eq_indexing):
        before = words_between[i] if i < len(words_between) else ""
        after  = words_between[i + 1] if i + 1 < len(words_between) else ""
        window = last_n_sents(before, 2) + " " + eq_alttexts[i] + " " + first_n_sents(after, 2)
        condensed_parts.append(window.strip())
    condensed = " ".join(condensed_parts)

    return True, total_text, eq_list, condensed, eq_indexing, num_to_id, id_to_num

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

BASE_INSTR = (
    "Analyze the context of the article to identify which equations are derived "
    "from each equation. Return a JSON object where keys are equation numbers and "
    "values are lists of equation numbers derived from that equation."
)

def build_prompt(variant, total_text, eq_list, condensed):
    """Returns user_prompt string. Structured output enforces JSON format on all providers."""
    art_header = "I have the following article that contains various mathematical equations:\n"
    eq_header  = "\nFrom this article, I have extracted the list of equations as follows:\n"

    if variant == "P0":
        user = art_header + total_text + eq_header + eq_list + "\n" + BASE_INSTR

    elif variant == "P1":
        instr = (
            "For each equation, identify which earlier equations it directly depends on "
            "or was built from. Return a JSON object where keys are equation numbers and "
            "values are lists of equation numbers that the key equation was derived from."
        )
        user = art_header + total_text + eq_header + eq_list + "\n" + instr

    elif variant == "P2":
        instr = (
            "Construct a directed acyclic graph where each directed edge (i -> j) means "
            "equation j could not have been written without equation i. Return a JSON "
            "object where keys are source equation numbers and values are lists of "
            "target equation numbers."
        )
        user = art_header + total_text + eq_header + eq_list + "\n" + instr

    elif variant == "P3":
        instr = (
            "First reason about the mathematical flow of the article and which results "
            "build on which. Then return a JSON object with two fields: reasoning (your "
            "chain-of-thought as a string) and derivation_graph (keys are equation "
            "numbers, values are lists of derived equation numbers)."
        )
        user = art_header + total_text + eq_header + eq_list + "\n" + instr

    elif variant == "P4":
        definition = (
            "An equation B is derived from equation A if: A defines a term used in B, "
            "A is explicitly substituted into B, or B is obtained by algebraically "
            "manipulating A. Only mark edges where this relationship clearly holds.\n"
        )
        user = art_header + total_text + eq_header + eq_list + "\n" + definition + BASE_INSTR

    elif variant == "P5":
        negative = (
            "Do NOT mark an edge simply because two equations share variables or "
            "notation. Only mark an edge if one equation is explicitly used to produce "
            "the other in the text of the article.\n"
        )
        user = art_header + total_text + eq_header + eq_list + "\n" + negative + BASE_INSTR

    elif variant == "P6":
        user = (
            "I have extracted the list of equations from a mathematical article as follows:\n"
            + eq_list + "\n" + BASE_INSTR
        )

    elif variant == "P7":
        user = (
            "I have the following condensed context from a mathematical article "
            "(2 sentences around each equation):\n"
            + condensed
            + "\nFrom this article, I have extracted the list of equations as follows:\n"
            + eq_list + "\n" + BASE_INSTR
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return user

# ---------------------------------------------------------------------------
# Truncation (keep beginning + end, cut middle)
# ---------------------------------------------------------------------------

def maybe_truncate(text, max_tokens, article_id, trunc_log):
    n = count_tokens(text)
    if n <= max_tokens:
        return text
    chars_per_tok = len(text) / max(n, 1)
    keep_chars = int(max_tokens * chars_per_tok * 0.9)
    half = keep_chars // 2
    result = text[:half] + "\n...[TRUNCATED]...\n" + text[-half:]
    trunc_log.append({
        "article_id": article_id,
        "original_tokens": n,
        "truncated_tokens": count_tokens(result),
    })
    return result

# ---------------------------------------------------------------------------
# Retry with exponential backoff
# ---------------------------------------------------------------------------

def call_with_retry(fn, max_retries=5):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                wait = (2 ** attempt) * (1.0 + random.random())
                print(f"    [retry {attempt+1}/{max_retries-1}] {type(e).__name__}: sleeping {wait:.1f}s")
                time.sleep(wait)
    raise last_exc or RuntimeError("call_with_retry: no attempts made")

# ---------------------------------------------------------------------------
# Per-provider call functions
# ---------------------------------------------------------------------------

def _cost(model_name, in_tok, out_tok):
    cfg = MODEL_CONFIG[model_name]
    return (in_tok * cfg["cost_input_per_1m"] + out_tok * cfg["cost_output_per_1m"]) / 1_000_000


def call_anthropic(model_name, user_prompt):
    cfg    = MODEL_CONFIG[model_name]
    schema = _COT_SCHEMA if _is_cot_active() else DERIVATION_GRAPH_SCHEMA
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    def _call():
        return client.messages.create(
            model=cfg["model_id"],
            max_tokens=4096,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            },
        )
    resp    = call_with_retry(_call)
    raw     = resp.content[0].text
    in_tok  = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens
    return raw, in_tok, out_tok, _cost(model_name, in_tok, out_tok), "anthropic"


def call_openai(model_name, user_prompt):
    cfg    = MODEL_CONFIG[model_name]
    schema = _COT_SCHEMA if _is_cot_active() else DERIVATION_GRAPH_SCHEMA
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    def _call():
        return client.chat.completions.create(
            model=cfg["model_id"],
            messages=[{"role": "user", "content": user_prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "derivation_graph", "schema": schema},
            },
        )
    resp    = call_with_retry(_call)
    raw     = resp.choices[0].message.content
    in_tok  = resp.usage.prompt_tokens
    out_tok = resp.usage.completion_tokens
    return raw, in_tok, out_tok, _cost(model_name, in_tok, out_tok), "openai"


def call_google(model_name, user_prompt):
    cfg    = MODEL_CONFIG[model_name]
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    def _call():
        return client.models.generate_content(
            model=cfg["model_id"],
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
    resp    = call_with_retry(_call)
    raw     = resp.text
    um      = getattr(resp, "usage_metadata", None)
    in_tok  = um.prompt_token_count     if um else count_tokens(user_prompt)
    out_tok = um.candidates_token_count if um else count_tokens(raw)
    return raw, in_tok, out_tok, _cost(model_name, in_tok, out_tok), "google"


def call_huggingface(model_name, user_prompt):
    cfg          = MODEL_CONFIG[model_name]
    providers    = cfg["hf_providers"]
    use_thinking = cfg.get("use_thinking", False)
    schema       = _COT_SCHEMA if _is_cot_active() else DERIVATION_GRAPH_SCHEMA
    last_exc     = None
    for provider in providers:
        try:
            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=os.environ["HF_TOKEN"],
            )
            model_id = cfg["model_id"]
            if "Qwen3" in model_id:
                messages = [{"role": "system", "content": "/no_think"},
                            {"role": "user",   "content": user_prompt}]
            else:
                messages = [{"role": "user", "content": user_prompt}]
            def _call(c=client, prov=provider, msgs=messages):
                return c.chat.completions.create(
                    model=f"{cfg['model_id']}:{prov}",
                    messages=msgs,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "derivation_graph", "schema": schema},
                    },
                    max_tokens=8192,
                )
            resp    = call_with_retry(_call)
            msg     = resp.choices[0].message
            raw     = msg.content or getattr(msg, "reasoning_content", None) or ""
            raw     = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            usage   = getattr(resp, "usage", None)
            in_tok  = usage.prompt_tokens     if usage else count_tokens(user_prompt)
            out_tok = usage.completion_tokens if usage else count_tokens(raw)
            return raw, in_tok, out_tok, _cost(model_name, in_tok, out_tok), provider
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["rate limit", "429", "unavailable", "503", "not found",
                                       "model", "quota", "overload"]):
                print(f"    [{SHORT_NAME[model_name]}] provider {provider} failed ({type(e).__name__}), trying next")
                last_exc = e
                continue
            raise
    raise RuntimeError(f"All HF providers failed for {model_name}: {last_exc}")


# variant context for schema selection inside call_* functions
_current_variant: str = "P0"

def _is_cot_active() -> bool:
    return _current_variant == "P3"


def call_model(model_name, user_prompt, clients, variant="P0"):
    global _current_variant
    _current_variant = variant
    provider = MODEL_CONFIG[model_name]["provider"]
    if provider == "anthropic":
        return call_anthropic(model_name, user_prompt)
    elif provider == "openai":
        return call_openai(model_name, user_prompt)
    elif provider == "google":
        return call_google(model_name, user_prompt)
    elif provider == "hf":
        return call_huggingface(model_name, user_prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")

# ---------------------------------------------------------------------------
# Output parsing (3-strategy cascade)
# ---------------------------------------------------------------------------

def parse_output(raw_output):
    """
    Returns (graph_dict_or_None, strategy_int, fail_reason_or_None)
    graph_dict uses number strings as keys: {"1": ["2", "3"], ...}
    """
    def _extract_wrapped(text):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("derivation_graph"), dict):
                return parsed["derivation_graph"]
        except Exception:
            pass
        return None

    def _extract_flat(text):
        # Gemini returns a flat {"1": [...], "2": [...]} without the wrapper
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and all(
                isinstance(v, list) for v in parsed.values()
            ):
                return parsed
        except Exception:
            pass
        return None

    # Strategy 1: wrapped {"derivation_graph": {...}}
    r = _extract_wrapped(raw_output)
    if r is not None:
        return r, 1, None

    # Strategy 2: brute-force find outermost JSON object, try wrapped then flat
    s, e = raw_output.find("{"), raw_output.rfind("}")
    if s != -1 and e > s:
        chunk = raw_output[s : e + 1]
        r = _extract_wrapped(chunk)
        if r is not None:
            return r, 2, None
        r = _extract_flat(chunk)
        if r is not None:
            return r, 2, None

    # Strategy 3: flat on full text
    r = _extract_flat(raw_output)
    if r is not None:
        return r, 3, None

    return None, 3, raw_output[:500]

# ---------------------------------------------------------------------------
# Equation ID mapping
# ---------------------------------------------------------------------------

def map_numbers_to_ids(graph_numbers, num_to_id, article_id):
    """Returns (graph_ids, mapping_errors_list)."""
    graph_ids = {}
    errors    = []
    for src_num, tgt_nums in graph_numbers.items():
        src_id = num_to_id.get(str(src_num))
        if src_id is None:
            errors.append({"article_id": article_id, "num": src_num, "role": "src"})
            continue
        tgt_ids = []
        for t in (tgt_nums or []):
            tid = num_to_id.get(str(t))
            if tid is None:
                errors.append({"article_id": article_id, "num": t, "role": "tgt"})
            else:
                tgt_ids.append(tid)
        graph_ids[src_id] = tgt_ids
    return graph_ids, errors


def invert_edges(graph_ids):
    """Invert all edges. Used post-processing for P1."""
    all_nodes = set(graph_ids.keys())
    for tgts in graph_ids.values():
        all_nodes.update(tgts)
    inv = {n: [] for n in all_nodes}
    for src, tgts in graph_ids.items():
        for tgt in tgts:
            inv[tgt].append(src)
    return inv

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_article_tp_fp_fn(true_adj, pred_adj):
    true_edges = {
        (src, tgt)
        for src, tgts in true_adj.items()
        for tgt in (tgts or [])
        if tgt is not None
    }
    pred_edges = {
        (src, tgt)
        for src, tgts in pred_adj.items()
        for tgt in (tgts or [])
        if tgt is not None
    }
    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    return tp, fp, fn, len(true_edges)


def pooled_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def macro_f1(per_article):
    """per_article: list of (tp, fp, fn)."""
    ps, rs, fs = [], [], []
    for tp, fp, fn in per_article:
        p, r, f = pooled_f1(tp, fp, fn)
        ps.append(p); rs.append(r); fs.append(f)
    n = len(fs) or 1
    return sum(ps)/n, sum(rs)/n, sum(fs)/n

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _safe_id(article_id):
    return article_id.replace("/", "_")


def save_article_result(out_dir, article_id, raw_output, parsed_edges, metrics):
    out_dir.mkdir(parents=True, exist_ok=True)
    sid = _safe_id(article_id)
    (out_dir / f"{sid}_raw_output.txt").write_text(raw_output or "", encoding="utf-8")
    (out_dir / f"{sid}_parsed_edges.json").write_text(json.dumps(parsed_edges, indent=2), encoding="utf-8")
    (out_dir / f"{sid}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def result_exists(out_dir, article_id):
    return (out_dir / f"{_safe_id(article_id)}_metrics.json").exists()


def load_metrics(out_dir, article_id):
    path = out_dir / f"{_safe_id(article_id)}_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Core: run one article through one model
# ---------------------------------------------------------------------------

def run_one_article(article_id, article_data, html_content,
                    model_name, variant, clients, out_dir,
                    trunc_log, failures):
    """
    Returns a metrics dict, or None on unrecoverable error.
    Saves files to out_dir immediately; is resume-safe.
    """
    if result_exists(out_dir, article_id):
        return load_metrics(out_dir, article_id)

    ok, total_text, eq_list, condensed, _, num_to_id, _ = \
        build_article_inputs(article_id, article_data, html_content)

    if not ok:
        failures.append({"article_id": article_id, "model": model_name,
                         "variant": variant, "error": "equation_set_mismatch"})
        return None

    # Truncate if needed (leave 2k tokens headroom for prompt overhead)
    ctx = MODEL_CONFIG[model_name]["context_limit_tokens"]
    total_text = maybe_truncate(total_text, ctx - 2000, article_id, trunc_log)

    user_prompt = build_prompt(variant, total_text, eq_list, condensed)

    # API call
    try:
        raw, in_tok, out_tok, cost_usd, provider_used = \
            call_model(model_name, user_prompt, clients, variant)
    except Exception as e:
        failures.append({"article_id": article_id, "model": model_name,
                         "variant": variant, "error": f"api: {e}"})
        m = {"article_id": article_id, "parse_failed": True,
             "error": str(e), "input_tokens": 0, "output_tokens": 0,
             "cost_usd": 0.0, "provider_used": "failed",
             "tp": 0, "fp": 0, "fn": 0, "n_predicted_edges": 0, "mapping_errors": []}
        save_article_result(out_dir, article_id, str(e), {}, m)
        return m

    # Parse
    graph_numbers, strategy, fail_reason = parse_output(raw)

    if graph_numbers is None:
        failures.append({"article_id": article_id, "model": model_name,
                         "variant": variant, "parse_strategy": 3,
                         "raw_output": raw[:1000]})
        m = {"article_id": article_id, "parse_failed": True,
             "parse_strategy": 3, "fail_reason": fail_reason,
             "input_tokens": in_tok, "output_tokens": out_tok,
             "cost_usd": cost_usd, "provider_used": provider_used,
             "tp": 0, "fp": 0, "fn": 0, "n_predicted_edges": 0, "mapping_errors": []}
        save_article_result(out_dir, article_id, raw, {}, m)
        return m

    # Map numbers → IDs
    graph_ids, map_errors = map_numbers_to_ids(graph_numbers, num_to_id, article_id)

    # P1: save pre-inversion, then invert
    pre_inv = None
    if variant == "P1":
        pre_inv = {k: list(v) for k, v in graph_ids.items()}
        graph_ids = invert_edges(graph_ids)

    tp, fp, fn, _ = compute_article_tp_fp_fn(article_data["Adjacency List"], graph_ids)
    n_pred       = sum(len(v) for v in graph_ids.values())
    n_pred_eq    = len(graph_ids)
    n_true_eq    = len(article_data.get("Equation ID", []))
    all_empty    = n_pred_eq > 0 and all(len(v) == 0 for v in graph_ids.values())
    n_hallucinated = len({e["num"] for e in map_errors})

    m = {
        "article_id":       article_id,
        "parse_failed":     False,
        "parse_strategy":   strategy,
        "tp": tp, "fp": fp, "fn": fn,
        "input_tokens":     in_tok,
        "output_tokens":    out_tok,
        "cost_usd":         cost_usd,
        "provider_used":    provider_used,
        "mapping_errors":   map_errors,
        "n_predicted_edges": n_pred,
        "n_predicted_equations": n_pred_eq,
        "n_true_equations":      n_true_eq,
        "full_empty_prediction":    all_empty and n_pred_eq == n_true_eq,
        "partial_empty_prediction": all_empty and n_pred_eq < n_true_eq,
        "n_hallucinated_nodes":     n_hallucinated,
        "pre_inversion_edges": pre_inv,
    }

    edges_out = {"predicted": {k: list(v) for k, v in graph_ids.items()}}
    if pre_inv:
        edges_out["pre_inversion"] = pre_inv
    save_article_result(out_dir, article_id, raw, edges_out, m)
    return m

# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------

def init_clients(models):
    clients = {}
    for model_name in models:
        cfg  = MODEL_CONFIG[model_name]
        prov = cfg["provider"]
        if prov == "anthropic" and "anthropic" not in clients:
            clients["anthropic"] = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif prov == "openai" and "openai" not in clients:
            clients["openai"] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        elif prov == "google" and "google" not in clients:
            clients["google"] = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        # HF reads HF_TOKEN directly in call_hf
    return clients


def check_env(models):
    missing = []
    env_map = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
               "google": "GOOGLE_API_KEY", "hf": "HF_TOKEN"}
    seen = set()
    for model_name in models:
        prov = MODEL_CONFIG[model_name]["provider"]
        key  = env_map[prov]
        if key not in seen and key not in os.environ:
            missing.append(f"  export {key}   # for {SHORT_NAME[model_name]}")
            seen.add(key)
    return missing

# ---------------------------------------------------------------------------
# Stage 0 — fix 20-article subset
# ---------------------------------------------------------------------------

def stage0(all_article_ids):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "prompt_selection_articles.json"

    if path.exists():
        data   = json.loads(path.read_text())
        subset = data["selection_subset"]
        print(f"[Stage 0] Subset already exists: {path}")
        print(f"  {subset}")
        return subset

    random.seed(42)
    subset = random.sample(all_article_ids, 20)
    print("[Stage 0] Proposed 20-article subset:")
    for aid in subset:
        print(f"  {aid}")
    ans = input("Save this subset? [y/N]: ").strip().lower()
    if ans != "y":
        print("Aborted.")
        sys.exit(0)
    path.write_text(json.dumps({"selection_subset": subset}, indent=2))
    print(f"[Stage 0] Saved → {path}")
    return subset

# ---------------------------------------------------------------------------
# Stage 1 — prompt selection
# ---------------------------------------------------------------------------

def stage1(article_ids_dict, subset_ids, clients, models):
    STAGE1_DIR.mkdir(parents=True, exist_ok=True)

    # accumulators: variant → model → running totals
    acc = {v: {m: {"tp": 0, "fp": 0, "fn": 0, "parse_fail": 0, "n": 0,
                    "cost": 0.0, "map_err": 0, "n_pred": 0,
                    "n_full_empty": 0, "n_partial_empty": 0,
                    "n_hallucinated": 0, "n_pred_eq_sum": 0, "n_true_eq_sum": 0}
               for m in models}
           for v in VARIANTS}

    failures   = []
    trunc_log   = []
    total_cost  = 0.0
    total_calls = len(VARIANTS) * len(models) * len(subset_ids)
    call_n      = 0
    t0          = time.time()

    try:
        for variant in VARIANTS:
            for model_name in models:
                for article_id in subset_ids:
                    call_n += 1
                    out_dir = STAGE1_DIR / variant / model_name.replace("/", "_")

                    html_path = Path(f"articles/{article_id.replace('/', '_')}.html")
                    if not html_path.exists():
                        print(f"  [SKIP] {html_path} not found")
                        continue

                    elapsed = time.time() - t0
                    if call_n > 1:
                        eta_s = elapsed / (call_n - 1) * (total_calls - call_n + 1)
                        eta   = f"ETA {int(eta_s//3600):02d}h{int(eta_s%3600//60):02d}m"
                    else:
                        eta = "ETA --"
                    print(f"[S1 {call_n}/{total_calls}] {variant} | {SHORT_NAME.get(model_name, model_name)} | {article_id} | ${total_cost:.4f} | {eta}")

                    m = run_one_article(
                        article_id, article_ids_dict[article_id],
                        html_path.read_text(encoding="utf-8"),
                        model_name, variant, clients, out_dir,
                        trunc_log, failures,
                    )
                    if m is None:
                        continue

                    r = acc[variant][model_name]
                    r["n"] += 1
                    r["cost"] += m.get("cost_usd", 0.0)
                    total_cost += m.get("cost_usd", 0.0)
                    if m.get("parse_failed"):
                        r["parse_fail"] += 1
                    else:
                        r["tp"] += m["tp"]; r["fp"] += m["fp"]; r["fn"] += m["fn"]
                        r["n_pred"]          += m.get("n_predicted_edges", 0)
                        r["map_err"]         += len(m.get("mapping_errors", []))
                        r["n_full_empty"]    += int(m.get("full_empty_prediction", False))
                        r["n_partial_empty"] += int(m.get("partial_empty_prediction", False))
                        r["n_hallucinated"]  += m.get("n_hallucinated_nodes", 0)
                        r["n_pred_eq_sum"]   += m.get("n_predicted_equations", 0)
                        r["n_true_eq_sum"]   += m.get("n_true_equations", 0)
    except KeyboardInterrupt:
        print("\n[interrupted] Results saved. Re-run with --stage 1 to resume.")
        sys.exit(0)

    # Print results table
    short = [SHORT_NAME[m] for m in models]
    col_w = 11
    header = f"{'Variant':<10}" + "".join(f"{s:>{col_w}}" for s in short) + f"{'Mean F1':>10}"
    print("\n" + "=" * len(header))
    print("Stage 1 Results (pooled F1 per variant per model)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    variant_mean = {}
    for variant in VARIANTS:
        f1s = []
        row = f"{variant:<10}"
        for model_name in models:
            r = acc[variant][model_name]
            _, _, f1 = pooled_f1(r["tp"], r["fp"], r["fn"])
            f1s.append(f1)
            row += f"{f1:>{col_w}.4f}"
        mean = sum(f1s) / len(f1s) if f1s else 0.0
        variant_mean[variant] = mean
        row += f"{mean:>10.4f}"
        print(row)

    best_variant = max(variant_mean, key=lambda v: variant_mean[v])
    print(f"\nWinning variant: {best_variant}  (mean F1 = {variant_mean[best_variant]:.4f})")
    print(f"Total cost: ${total_cost:.4f}")

    # Save stage1 summary (timestamped — never overwrites)
    now1 = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = {
        "variants": VARIANTS, "models": models,
        "results": {v: acc[v] for v in VARIANTS},
        "variant_mean_f1": variant_mean,
        "winning_variant": best_variant,
        "total_cost_usd": total_cost,
        "truncations": trunc_log,
        "timestamp": now1,
    }
    (STAGE1_DIR / f"stage1_summary_{now1}.json").write_text(json.dumps(summary, indent=2))
    (STAGE1_DIR / f"stage1_failures_{now1}.json").write_text(json.dumps(failures, indent=2))

    # decisions.json
    decisions_path = RESULTS_DIR / "decisions.json"
    if decisions_path.exists():
        print(f"WARNING: {decisions_path} already exists — not overwriting.")
    else:
        ans = input(f"Write '{best_variant}' to decisions.json? [y/N]: ").strip().lower()
        if ans == "y":
            decisions_path.write_text(json.dumps({
                "winning_prompt_variant": best_variant,
                "winning_prompt_mean_f1": variant_mean[best_variant],
                "stage1_completed": datetime.date.today().isoformat(),
                "winning_temperature": "default",
                "notes": "",
            }, indent=2))
            print(f"Saved → {decisions_path}")

    return best_variant

# ---------------------------------------------------------------------------
# Stage 2 — final evaluation
# ---------------------------------------------------------------------------

def stage2(article_ids_dict, clients, winning_variant, models):
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    all_ids      = list(article_ids_dict.keys())
    n_runs       = 3
    total_cost   = 0.0
    combined     = {}
    total_calls  = len(models) * n_runs * len(all_ids)
    call_n       = 0
    t0           = time.time()

    for model_name in models:
        model_dir = STAGE2_DIR / model_name.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        trunc_log  = []
        failures   = []
        run_f1s    = []
        run_summaries = []

        for run_idx in range(1, n_runs + 1):
            run_dir = model_dir / f"run_{run_idx}"
            run_dir.mkdir(parents=True, exist_ok=True)

            tp_tot = fp_tot = fn_tot = 0
            per_art = []
            parse_fails = map_err_tot = n_pred_tot = 0
            n_full_empty = n_partial_empty = n_hallucinated = 0
            n_pred_eq_sum = n_true_eq_sum = 0
            cost_run = 0.0

            try:
                for article_id in all_ids:
                    call_n += 1
                    html_path = Path(f"articles/{article_id.replace('/', '_')}.html")
                    if not html_path.exists():
                        continue

                    elapsed = time.time() - t0
                    if call_n > 1:
                        eta_s = elapsed / (call_n - 1) * (total_calls - call_n + 1)
                        eta   = f"ETA {int(eta_s//3600):02d}h{int(eta_s%3600//60):02d}m"
                    else:
                        eta = "ETA --"
                    print(f"[S2 {call_n}/{total_calls}] {SHORT_NAME.get(model_name, model_name)} | run {run_idx}/{n_runs} | {article_id} | ${total_cost:.4f} | {eta}")

                    m = run_one_article(
                        article_id, article_ids_dict[article_id],
                        html_path.read_text(encoding="utf-8"),
                        model_name, winning_variant, clients, run_dir,
                        trunc_log, failures,
                    )
                    if m is None:
                        continue

                    cost_run  += m.get("cost_usd", 0.0)
                    total_cost += m.get("cost_usd", 0.0)

                    if m.get("parse_failed"):
                        parse_fails += 1
                    else:
                        tp, fp, fn = m["tp"], m["fp"], m["fn"]
                        tp_tot += tp; fp_tot += fp; fn_tot += fn
                        per_art.append((tp, fp, fn))
                        n_pred_tot      += m.get("n_predicted_edges", 0)
                        map_err_tot     += len(m.get("mapping_errors", []))
                        n_full_empty    += int(m.get("full_empty_prediction", False))
                        n_partial_empty += int(m.get("partial_empty_prediction", False))
                        n_hallucinated  += m.get("n_hallucinated_nodes", 0)
                        n_pred_eq_sum   += m.get("n_predicted_equations", 0)
                        n_true_eq_sum   += m.get("n_true_equations", 0)

            except KeyboardInterrupt:
                print("\n[interrupted] Partial run saved. Re-run with --stage 2 to resume.")
                sys.exit(0)

            pool_p, pool_r, pool_f = pooled_f1(tp_tot, fp_tot, fn_tot)
            mac_p,  mac_r,  mac_f  = macro_f1(per_art)

            if abs(pool_f - mac_f) > 0.05:
                print(f"  WARNING: pooled F1 ({pool_f:.4f}) vs macro F1 ({mac_f:.4f}) "
                      f"diverge by {abs(pool_f-mac_f)*100:.1f}% — {model_name} run {run_idx}")

            n_ok = len(per_art)
            rs = {
                "run": run_idx,
                "pooled":  {"precision": pool_p, "recall": pool_r, "f1": pool_f},
                "macro":   {"precision": mac_p,  "recall": mac_r,  "f1": mac_f},
                "parse_failure_rate": parse_fails / max(n_ok + parse_fails, 1),
                "mean_edges_predicted": n_pred_tot / max(n_ok, 1),
                "mapping_errors": map_err_tot,
                "cost_usd": cost_run,
                "diagnostics": {
                    "n_parse_failures":           parse_fails,
                    "n_full_empty_predictions":   n_full_empty,
                    "n_partial_empty_predictions": n_partial_empty,
                    "n_hallucinated_nodes":       n_hallucinated,
                    "mean_predicted_equations":   n_pred_eq_sum / max(n_ok, 1),
                    "mean_true_equations":        n_true_eq_sum / max(n_ok, 1),
                    "total_extra_edges":          fp_tot,
                    "total_missing_edges":        fn_tot,
                },
            }
            run_summaries.append(rs)
            run_f1s.append(pool_f)

        f1_mean = sum(run_f1s) / len(run_f1s) if run_f1s else 0.0
        f1_std  = (sum((f - f1_mean)**2 for f in run_f1s) / len(run_f1s)) ** 0.5 if run_f1s else 0.0
        total_cost_model = sum(r["cost_usd"] for r in run_summaries)

        summary = {
            "model": model_name,
            "stage": 2,
            "prompt_variant": winning_variant,
            "output_format": "json_structured",
            "temperature": "default",
            "n_runs": n_runs,
            "n_articles": len(all_ids),
            "pooled": {
                "f1_mean": f1_mean,
                "f1_std":  f1_std,
                "precision_mean": sum(r["pooled"]["precision"] for r in run_summaries) / len(run_summaries),
                "recall_mean":    sum(r["pooled"]["recall"]    for r in run_summaries) / len(run_summaries),
            },
            "macro": {
                "f1_mean":        sum(r["macro"]["f1"]        for r in run_summaries) / len(run_summaries),
                "precision_mean": sum(r["macro"]["precision"] for r in run_summaries) / len(run_summaries),
                "recall_mean":    sum(r["macro"]["recall"]    for r in run_summaries) / len(run_summaries),
            },
            "parse_failure_rate":   sum(r["parse_failure_rate"] for r in run_summaries) / len(run_summaries),
            "mean_edges_predicted": sum(r["mean_edges_predicted"] for r in run_summaries) / len(run_summaries),
            "mean_edges_ground_truth": 7.31,
            "approx_cost_per_article_usd": total_cost_model / max(len(all_ids) * n_runs, 1),
            "total_cost_usd": total_cost_model,
            "diagnostics": {
                "n_parse_failures":           sum(r["diagnostics"]["n_parse_failures"]           for r in run_summaries),
                "n_full_empty_predictions":   sum(r["diagnostics"]["n_full_empty_predictions"]   for r in run_summaries),
                "n_partial_empty_predictions": sum(r["diagnostics"]["n_partial_empty_predictions"] for r in run_summaries),
                "n_hallucinated_nodes":       sum(r["diagnostics"]["n_hallucinated_nodes"]       for r in run_summaries),
                "mean_predicted_equations":   sum(r["diagnostics"]["mean_predicted_equations"]   for r in run_summaries) / len(run_summaries),
                "mean_true_equations":        sum(r["diagnostics"]["mean_true_equations"]        for r in run_summaries) / len(run_summaries),
                "total_extra_edges":          sum(r["diagnostics"]["total_extra_edges"]          for r in run_summaries),
                "total_missing_edges":        sum(r["diagnostics"]["total_missing_edges"]        for r in run_summaries),
            },
            "runs": run_summaries,
            "truncations": trunc_log,
        }
        now2 = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        summary["timestamp"] = now2
        (model_dir / f"summary_{now2}.json").write_text(json.dumps(summary, indent=2))
        (model_dir / f"failures_{now2}.json").write_text(json.dumps(failures, indent=2))
        combined[model_name] = summary

        print(f"\n  {SHORT_NAME[model_name]}: F1 = {f1_mean:.4f} ± {f1_std:.4f}  "
              f"(cost ${total_cost_model:.2f})")

    now2c = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (STAGE2_DIR / f"stage2_combined_summary_{now2c}.json").write_text(json.dumps(combined, indent=2))

    # Final table
    print("\n" + "=" * 90)
    print("Stage 2 Final Results")
    print("=" * 90)
    hdr = f"{'Model':<40} {'P':>7} {'R':>7} {'F1':>7} {'±':>7} {'Fail%':>6} {'$/art':>8}"
    print(hdr)
    print("-" * 90)
    for model_name, s in combined.items():
        p   = s["pooled"]["precision_mean"]
        r   = s["pooled"]["recall_mean"]
        f   = s["pooled"]["f1_mean"]
        std = s["pooled"]["f1_std"]
        fr  = s["parse_failure_rate"] * 100
        ca  = s["approx_cost_per_article_usd"]
        print(f"{model_name:<40} {p:>7.4f} {r:>7.4f} {f:>7.4f} {std:>7.4f} {fr:>6.1f} ${ca:>7.4f}")

# ---------------------------------------------------------------------------
# Cost upper bound estimation
# ---------------------------------------------------------------------------

def cost_estimate(article_ids_dict, clients, models):
    print("[Cost Estimate] Scanning articles for max token count...")

    max_tok = 0
    max_id  = None
    for article_id, article_data in article_ids_dict.items():
        html_path = Path(f"articles/{article_id.replace('/', '_')}.html")
        if not html_path.exists():
            continue
        try:
            ok, total_text, eq_list, *_ = build_article_inputs(
                article_id, article_data, html_path.read_text(encoding="utf-8")
            )
            if not ok:
                continue
        except Exception:
            continue
        tok = count_tokens((total_text or "") + "\n" + (eq_list or ""))
        if tok > max_tok:
            max_tok = tok; max_id = article_id

    if max_id is None:
        print("No articles found.")
        return

    print(f"  Most expensive article: {max_id}  ({max_tok} tokens)")
    ans = input("Run this article through all models with P0? [y/N]: ").strip().lower()
    if ans != "y":
        sys.exit(0)

    html_path    = Path(f"articles/{max_id.replace('/', '_')}.html")
    article_data = article_ids_dict[max_id]
    ok2, total_text, eq_list, condensed, *_ = build_article_inputs(
        max_id, article_data, html_path.read_text(encoding="utf-8")
    )
    if not ok2 or total_text is None or eq_list is None or condensed is None:
        print("Failed to parse most expensive article.")
        return
    user_prompt = build_prompt("P0", total_text, eq_list, condensed)

    now_ce = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ce_dir = RESULTS_DIR / "cost_estimate" / now_ce
    ce_dir.mkdir(parents=True, exist_ok=True)

    col = 46
    print(f"\n{'Model':<{col}} {'In':>8} {'Out':>8} {'1x $':>10} {'S1 UB':>12} {'S2 UB':>12} Provider")
    print("-" * 110)

    s1_total = s2_total = 0.0
    cost_rows = []
    for model_name in models:
        try:
            raw, in_tok, out_tok, cost1x, provider = \
                call_model(model_name, user_prompt, clients, "P0")
            s1 = cost1x * 64 * 8   # 64 articles × 8 variants
            s2 = cost1x * 64 * 3   # 64 articles × 3 runs
            s1_total += s1; s2_total += s2
            print(f"{model_name:<{col}} {in_tok:>8} {out_tok:>8} ${cost1x:>9.4f} ${s1:>11.2f} ${s2:>11.2f} {provider}")
            sid = model_name.replace("/", "_")
            (ce_dir / f"{sid}_raw.txt").write_text(raw or "", encoding="utf-8")
            graph, strategy, _ = parse_output(raw or "")
            (ce_dir / f"{sid}_parsed.json").write_text(json.dumps({"parsed": graph, "strategy": strategy}, indent=2))
            cost_rows.append({"model": model_name, "provider": provider,
                               "input_tokens": in_tok, "output_tokens": out_tok,
                               "cost_1x": cost1x, "s1_ub": s1, "s2_ub": s2})
        except Exception as e:
            print(f"{model_name:<{col}} ERROR: {e}")
            cost_rows.append({"model": model_name, "error": str(e)})

    (ce_dir / "cost_summary.json").write_text(json.dumps({
        "article_id": max_id, "input_tokens_max": max_tok,
        "s1_total_ub": s1_total, "s2_total_ub": s2_total,
        "rows": cost_rows,
    }, indent=2))

    print("-" * 110)
    print(f"{'TOTAL':.<{col}} {'':>8} {'':>8} {'':>10} ${s1_total:>11.2f} ${s2_total:>11.2f}")
    print(f"\nStage 1 upper bound: ${s1_total:.2f}  (64 articles × 8 variants, all models)")
    print(f"Stage 2 upper bound: ${s2_total:.2f}  (64 articles × 3 runs, all models)")
    print(f"Grand total upper bound: ${s1_total + s2_total:.2f}")

    ans2 = input("\nProceed with pipeline stages? [y/N]: ").strip().lower()
    if ans2 != "y":
        print("Exiting.")
        sys.exit(0)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="LLM derivation graph extraction pipeline")
    p.add_argument("--stage", choices=["0", "1", "2", "all"], default="all")
    p.add_argument("--cost-estimate", action="store_true",
                   help="Find most expensive article, run through all models, print upper-bound table")
    p.add_argument("--models", nargs="+", default=None,
                   help="Subset of models to run (default: all 6)")
    args = p.parse_args()

    models = args.models if args.models else list(ALL_MODELS)
    for m in models:
        if m not in MODEL_CONFIG:
            p.error(f"Unknown model: {m}. Valid: {list(MODEL_CONFIG.keys())}")

    # Check env vars
    missing = check_env(models)
    if missing:
        print("Missing required environment variables:")
        for line in missing:
            print(line)
        sys.exit(1)

    clients          = init_clients(models)
    article_ids_dict = {k: v for k, v in article_parser.get_manually_parsed_articles().items()
                        if k not in BROKEN_ARTICLES}
    all_ids          = list(article_ids_dict.keys())

    if args.cost_estimate:
        cost_estimate(article_ids_dict, clients, models)
        return

    # Stage 0
    if args.stage in ("0", "all"):
        subset_ids = stage0(all_ids)
    else:
        sp = RESULTS_DIR / "prompt_selection_articles.json"
        if not sp.exists():
            print("ERROR: Run Stage 0 first (--stage 0).")
            sys.exit(1)
        subset_ids = json.loads(sp.read_text())["selection_subset"]

    # Stage 1
    winning_variant = None
    if args.stage in ("1", "all"):
        winning_variant = stage1(article_ids_dict, subset_ids, clients, models)

    # Stage 2
    if args.stage in ("2", "all"):
        if winning_variant is None:
            dp = RESULTS_DIR / "decisions.json"
            if not dp.exists():
                print("ERROR: Run Stage 1 first or create decisions.json.")
                sys.exit(1)
            winning_variant = json.loads(dp.read_text())["winning_prompt_variant"]
        stage2(article_ids_dict, clients, winning_variant, models)


if __name__ == "__main__":
    main()
