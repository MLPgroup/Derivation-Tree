# LLM Experiment Runner for MDGD Derivation Graph Extraction

## Overview

Three-stage experiment pipeline:

    Stage 1   - Prompt Selection        (20-article subset, all models, 1 run)
    Stage 2   - Final Evaluation        (all 64 articles, all models, 3 runs)

The 20-article subset is sampled once with a fixed seed and reused across
Stages 1. The same articles appear in Stage 2 — this is acceptable
because no parameters are fit during prompt.

All experiments use JSON structured output exclusively. The task output is
an adjacency list (structured data by definition), and free-form output
introduces parse failures as a confound unrelated to mathematical understanding.

All models run at their default temperature. This ensures results reflect out-of-the-box
performance and removes temperature as a confounding variable in cross-model
comparisons.

---

## Dataset

    Total articles:         64
    Total equation nodes:   665
    Total directed edges:   468 (ground truth)
    Mean edges per article: 7.31
    Min edges per article:  1
    Max edges per article:  25
    Source: MDGD adjacency lists

---

## Ground Truth

- A predicted edge is a TP if (i -> j) exists in the MDGD adjacency list
- Edge matching is exact: both source and target equation IDs must match
- Direction matters: (i -> j) is not the same as (j -> i)

---

## Models

    Closed API models:
      claude-opus-4-6              Anthropic API   #1 Document / #2 Math (Arena May 2026)
      gpt-5                        OpenAI API      #1 Math / top-5 Document
      gemini-3.1-pro-preview       Google API      #4 Math

    Open models (HuggingFace Inference Providers):
      moonshotai/Kimi-K2.6         HuggingFace     #10 Document / #12 Math
      Qwen/Qwen2.5-Math-7B-Instruct  HuggingFace   math-specialised small model
      deepseek-ai/DeepSeek-R1-Distill-Qwen-7B  HuggingFace  open reasoning model

    Analytical baselines (carry over existing results, no rerunning needed):
      Brute Force
      Token String Similarity
      Naive Bayes (pooled, 5-fold cross-validation)

---

## JSON Schema

### Single-field schema (all variants except P3)

    {
      "derivation_graph": {
        "1": ["3", "5"],
        "2": ["3"],
        "3": [],
        "4": ["5"],
        "5": []
      }
    }

Keys are equation numbers as strings.
Values are lists of equation numbers the key equation directly derives into.
Empty list means no outgoing edges.
All equation numbers must appear as keys even if their value is an empty list.

### Two-field schema (P3 chain-of-thought only)

    {
      "reasoning": "equation 3 builds on equation 1 because...",
      "derivation_graph": {
        "1": ["3", "5"],
        ...
      }
    }

The reasoning field is ignored for metrics but retained for qualitative analysis.

---

## Structured Output Implementation Per Provider

### Anthropic (claude-opus-4-6)

    Use system prompt enforcing JSON only:
    system = "You are a mathematical reasoning assistant. You must respond with
              valid JSON only, following the exact schema provided. Do not include
              any text outside the JSON object."

### OpenAI (gpt-5)

    response_format={
      "type": "json_schema",
      "json_schema": {
        "name": "derivation_graph",
        "schema": {
          "type": "object",
          "properties": {
            "derivation_graph": {
              "type": "object",
              "additionalProperties": {
                "type": "array",
                "items": {"type": "string"}
              }
            }
          },
          "required": ["derivation_graph"]
        }
      }
    }

### Google (gemini-3.1-pro-preview)

    generation_config={"response_mime_type": "application/json"}

### HuggingFace Inference Providers (Kimi, Qwen, DeepSeek)

    from huggingface_hub import InferenceClient

    client = InferenceClient(provider="...", api_key="HF_TOKEN")
    response = client.chat.completions.create(
        model="...",
        messages=[...],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "derivation_graph",
                "schema": {
                    "type": "object",
                    "properties": {
                        "derivation_graph": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    },
                    "required": ["derivation_graph"]
                }
            }
        }
    )

---

## Prompt Variants

All variants use JSON structured output.
Replace only the instruction paragraph — keep article text and equation list
inputs identical across all variants.

### P0 - Baseline

    System: (see structured output implementation above)

    User:
    "I have the following article that contains various mathematical equations:\n"
    + {total_article_text}
    + "\nFrom this article, I have extracted the list of equations as follows:\n"
    + {equation_list}
    + "\nAnalyze the context of the article to identify which equations are derived
    from each equation. Return a JSON object where keys are equation numbers and
    values are lists of equation numbers derived from that equation."

### P1 - Dependency framing

    Replace instruction with:
    "For each equation, identify which earlier equations it directly depends on
    or was built from. Return a JSON object where keys are equation numbers and
    values are lists of equation numbers that the key equation was derived from."

    Note: this reverses the edge direction in the prompt. Post-process by
    inverting all edges before computing metrics. Log raw pre-inversion
    edges separately.

### P2 - Explicit DAG framing

    Replace instruction with:
    "Construct a directed acyclic graph where each directed edge (i -> j) means
    equation j could not have been written without equation i. Return a JSON
    object where keys are source equation numbers and values are lists of
    target equation numbers."

### P3 - Chain-of-thought

    Replace instruction with:
    "First reason about the mathematical flow of the article and which results
    build on which. Then return a JSON object with two fields: reasoning (your
    chain-of-thought as a string) and derivation_graph (keys are equation
    numbers, values are lists of derived equation numbers)."

    Use two-field JSON schema for this variant only.

### P4 - Definition of derivation provided

    Insert before the instruction:
    "An equation B is derived from equation A if: A defines a term used in B,
    A is explicitly substituted into B, or B is obtained by algebraically
    manipulating A. Only mark edges where this relationship clearly holds."

    Keep base instruction from P0.

### P5 - Negative instruction

    Insert before the instruction:
    "Do NOT mark an edge simply because two equations share variables or
    notation. Only mark an edge if one equation is explicitly used to produce
    the other in the text of the article."

    Keep base instruction from P0.

### P6 - Equations only, no article text

    Remove total_article_text entirely.
    Pass only equation_list.
    Keep instruction from P0.
    Tests whether full article context helps or hurts.

### P7 - Condensed context

    Replace total_article_text with condensed_context:
    For each equation, extract the 2 sentences immediately before and after it.
    Concatenate these windows in document order.
    Keep instruction from P0.

---

## Output Parsing

For JSON structured output, parsing should succeed for all well-behaved API
calls. Apply fallback strategies only when the model returns malformed JSON
despite structured output mode being enabled.

    Strategy 1: direct JSON parse
      json.loads(output)
      if "derivation_graph" key present and value is dict, use it

    Strategy 2: extract JSON substring
      find first { and last } in output
      attempt json.loads on that substring

    Strategy 3: declare failure
      log article_id and raw output to failures.json
      exclude from ALL metric calculations for that run
      do NOT substitute zeros

---

## Metrics

### Primary: pooled over all articles

Aggregate all TP, FP, FN across all articles then compute once:

    Precision = sum(TP) / (sum(TP) + sum(FP))
    Recall    = sum(TP) / (sum(TP) + sum(FN))
    F1        = 2 * Precision * Recall / (Precision + Recall)

Sums are over all articles, excluding parse failures.

Rationale: your dataset has substantial variation in graph size (1--25 edges
per article). Pooled metrics weight each edge equally regardless of which
article it came from. Macro-averaging would give equal weight to a 1-edge
article and a 25-edge article, which is inappropriate given this variation
and inconsistent with how the ground truth was constructed.

### Secondary: macro-averaged over articles (appendix only)

    Compute P, R, F1 per article.
    Take mean across all articles, excluding parse failures.
    Report in appendix alongside pooled for consistency check.
    If pooled and macro-averaged diverge by more than 5%, investigate
    which article sizes are driving the gap — this is itself a finding.

### Per-run tracking (3 runs per model in Stage 2)

    Report mean and std of pooled F1 across 3 runs.
    High std indicates model instability at default temperature.

### Additional columns per model

    parse_failure_rate          count(failed parses) / total outputs
    mean_edges_predicted        mean predicted edges per article
    mean_edges_ground_truth     7.31 (fixed reference)
    approx_cost_per_article     total API cost / number of articles

---

## Step 0 - Fix the 20-Article Subset

Run this once before doing anything else.
Save the output and never resample.

    import json
    import random

    random.seed(42)
    all_article_ids = [...]  # load your 64 article IDs here
    selection_subset = random.sample(all_article_ids, 20)

    with open("results/prompt_selection_articles.json", "w") as f:
        json.dump({"selection_subset": selection_subset}, f, indent=2)

All Stage 1 and Stage 1.5 runs load article IDs from this file.

---

## Stage 1 - Prompt Selection (20 articles, all models, 1 run)

### Setup

    Articles:     20-article subset
    Models:       all 6 LLM models
    Format:       JSON structured output
    Temperature:  model default for all models
    Runs:         1 per variant per model
    Variants:     P0 through P7

### Decision rule

For each variant, compute mean pooled F1 across all 6 models.
Select the variant with the highest mean pooled F1.
In case of tie, prefer the variant with the higher mean parse rate.
Record the winning variant in decisions.json before starting Stage 1.5.

### Stage 1 results table (fill in after running)

    Variant | claude | gpt-5 | gemini | kimi | qwen-math | deepseek | Mean F1
    P0      |        |       |        |      |           |          |
    P1      |        |       |        |      |           |          |
    P2      |        |       |        |      |           |          |
    P3      |        |       |        |      |           |          |
    P4      |        |       |        |      |           |          |
    P5      |        |       |        |      |           |          |
    P6      |        |       |        |      |           |          |
    P7      |        |       |        |      |           |          |

    Winning variant:
    Winning variant mean F1:
    Notes on per-model disagreement (if any):

---

## Stage 2 - Final Evaluation (all 64 articles, all models, 3 runs)

### Setup

    Articles:     all 64 articles
    Models:       all 6 LLM models
    Prompt:       winning variant from Stage 1
    Format:       JSON structured output
    Temperature:  model default
    Runs:         3 per model

### Stage 2 results table (fill in after running)

    Model                      | Precision | Recall | F1    | F1 Std | Fail Rate | Cost/Article
    Brute Force                |           |        |       | n/a    | n/a       | n/a
    Token Similarity           |           |        |       | n/a    | n/a       | n/a
    Naive Bayes (5-fold)       |           |        |       |        | n/a       | n/a
    claude-opus-4-6            |           |        |       |        |           |
    gpt-5                      |           |        |       |        |           |
    gemini-3.1-pro-preview     |           |        |       |        |           |
    Kimi-K2.6                  |           |        |       |        |           |
    Qwen2.5-Math-7B            |           |        |       |        |           |
    DeepSeek-R1-Distill        |           |        |       |        |           |

---

## Output File Structure

    results/
      prompt_selection_articles.json
      decisions.json
      stage1_prompt_selection/
        P0/
          {article_id}_raw_output.txt
          {article_id}_parsed_edges.json
          {article_id}_metrics.json
        P1/
        P2/
        P3/
        P4/
        P5/
        P6/
        P7/
        stage1_summary.json
        stage1_failures.json
      stage2_final/
        {model_name}/
          run_1/
            {article_id}_raw_output.txt
            {article_id}_parsed_edges.json
            {article_id}_metrics.json
          run_2/
          run_3/
          summary.json
          failures.json
        stage2_combined_summary.json

---

## decisions.json Schema

    {
      "winning_prompt_variant": "",
      "winning_prompt_mean_f1": 0.0,
      "stage1_completed": "YYYY-MM-DD",
      "winning_temperature": "default",
      "notes": ""
    }

Record after Stage 1. Update after Stage 1.5. Do not modify after Stage 2 begins.

---

## summary.json Schema (Stage 2)

    {
      "model": "claude-opus-4-6",
      "stage": 2,
      "prompt_variant": "P0",
      "output_format": "json_structured",
      "temperature": "default",
      "n_runs": 3,
      "n_articles": 64,
      "pooled": {
        "precision_mean": 0.0,
        "precision_std": 0.0,
        "recall_mean": 0.0,
        "recall_std": 0.0,
        "f1_mean": 0.0,
        "f1_std": 0.0
      },
      "macro": {
        "precision_mean": 0.0,
        "precision_std": 0.0,
        "recall_mean": 0.0,
        "recall_std": 0.0,
        "f1_mean": 0.0,
        "f1_std": 0.0
      },
      "parse_failure_rate": 0.0,
      "mean_edges_predicted": 0.0,
      "mean_edges_ground_truth": 7.31,
      "approx_cost_per_article_usd": 0.0
    }

---

## Implementation Notes

- Run Step 0 first. Fix the 20-article subset before running anything. Never
  resample.
- Record decisions.json after each stage before starting the next. Never
  modify it after Stage 2 begins.
- Failed parses: log and exclude from all metrics. Never substitute zeros.
- Context length: if article exceeds model context limit, truncate from the
  middle (keep beginning and end). Log all truncations with article_id,
  original token count, and truncated token count.
- Rate limiting: exponential backoff, max 5 retries, then log as failure.
- Token usage: record input and output tokens per call for cost tracking.
- P1 edge direction: post-process P1 outputs by inverting all edges before
  computing metrics. Log raw pre-inversion edges separately.
- The 20 selection articles are included in the final 64-article evaluation.
  This is acceptable because no parameters are fit during selection — only
  an instruction is chosen.
- If pooled F1 and macro-averaged F1 diverge by more than 5% for any model,
  flag this in the summary and investigate which article sizes drive the gap.