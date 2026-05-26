# Claude Code Task: Build a Fine-Tuned Encoder Baseline for the MDGD Derivation Graph Extraction Task

## Context

I'm building a fine-tuned encoder baseline for a paper on extracting mathematical derivation graphs from STEM articles. The dataset (MDGD) consists of 64 arXiv articles, each with a set of equations and hand-annotated directed edges between them representing derivation relationships. The task is binary classification on ordered equation pairs: given two equations (vi, vj) where i < j, predict whether there's a directed edge vi → vj.

I need a Google Colab notebook that fine-tunes encoder models (SciBERT, DeBERTa-v3) on this task and reports edge-level precision/recall/F1 matching the protocol used in the rest of the paper.

## Deliverable

A single self-contained Jupyter notebook (`mdgd_encoder_baseline.ipynb`) that I can upload to Google Colab and run end-to-end on a free T4 GPU. The notebook should:

1. Install dependencies
2. Mount Google Drive or accept dataset upload
3. Load and preprocess the MDGD dataset
4. Construct training pairs with proper features
5. Fine-tune an encoder with 5-fold cross-validation
6. Compute pooled edge-set metrics (precision, recall, F1) matching the paper's protocol
7. Save results and per-fold predictions to a results JSON
8. Be configurable to run different encoder models with a single parameter change

## Dataset Specification

The dataset is a JSON file with the following structure (one entry per article):

```json
{
  "Article ID": "1409.0466",
  "Equation ID": ["S3.E1", "S3.E2", "S3.E3", "..."],
  "Adjacency List": {
    "S3.E1": ["S3.E3", "S3.E5"],
    "S3.E2": ["S3.E3"],
    "S3.E3": [null]
  },
  "Equation Number": {"S3.E1": "1", "S3.E2": "2"}
}
```

The adjacency list maps each equation to a list of equations it derives (outgoing edges). A `[null]` or empty list means no outgoing edges.

**For each article, I also have the raw HTML from ar5iv.** Assume this is provided as a separate file or directory keyed by Article ID. The HTML contains the equations (as MathML), surrounding prose, and hyperlinked equation references.

The notebook should accept the dataset as:

- `mdgd_dataset.json` — the adjacency lists for all 64 articles
- `articles/` — a directory containing `{Article ID}.html` for each article

If HTML files aren't provided locally, the notebook should download them from `https://ar5iv.org/abs/{article_id}` and cache them in a local `articles/` directory. If a download fails, fall back gracefully to using only equation IDs and a placeholder context (and warn loudly that performance will be degraded).

## Pair Construction Protocol

For each article with equations [v1, v2, ..., vn] in document order:

1. Generate all ordered pairs (vi, vj) with i < j. An article with n equations contributes n(n−1)/2 candidate pairs.
2. Label each pair as **1** if there's a directed edge vi → vj in the adjacency list, else **0**.
3. Note: this is a binary task (edge / no edge), not the three-class formulation used by Naive Bayes. The directed edge direction is fixed by document order.

This matches the protocol described in §6 of the paper: pooled edge-set metrics over directed edges, no true-negative term.

## Feature Construction

For each ordered pair (vi, vj), construct an input string of the form:

```
[CLS] {equation_i_text} [SEP] {context_between} [SEP] {equation_j_text} [SEP]
```

Where:

- `equation_i_text` and `equation_j_text` are the MathML alt-text or LaTeX representations of the equations (extracted from the HTML).
- `context_between` is a windowed slice of the article text. **Use the following window strategy:**
  - The last 2 sentences before equation vi
  - All text between vi and vj, **truncated to the middle if too long**
  - The first 2 sentences after vj
- Truncate the full input to the model's max length (default 512 tokens), prioritizing the equation texts over the context (i.e., keep both equations fully and truncate the middle context).

Also add two scalar features that get concatenated to the [CLS] embedding before the classifier head:

- **Equation distance**: j − i (normalized by total equation count)
- **Same paragraph indicator**: 1 if the two equations are in the same paragraph in the HTML, else 0

These positional features matter a lot for this task and shouldn't be discarded.

## Model Architecture

Use HuggingFace Transformers. Build a custom classification head:

```
encoder (SciBERT / DeBERTa-v3) → [CLS] embedding (768-dim)
                                       ↓
                                  concatenate scalar features (2-dim)
                                       ↓
                                  Linear(770 → 256) + ReLU + Dropout(0.1)
                                       ↓
                                  Linear(256 → 2)  [logits for binary classification]
```

Make the encoder choice a top-of-notebook parameter:

```python
MODEL_NAME = "allenai/scibert_scivocab_uncased"  # or "microsoft/deberta-v3-base"
```

## Training Configuration

- **Optimizer**: AdamW, learning rate 2e-5 for encoder, 1e-4 for classifier head
- **Batch size**: 16 (reduce to 8 if OOM on T4)
- **Epochs**: 5 (with early stopping on val F1, patience 2)
- **Max sequence length**: 512
- **Loss**: Cross-entropy with **class weighting** — positive class weight should be ~15 (since positive rate is ~6%). Make this a tunable parameter.
- **Mixed precision** (`fp16=True`) for T4 compatibility and speed

## Evaluation Protocol

This must exactly match the protocol used elsewhere in the paper:

1. **5-fold cross-validation** at the **article level** (not pair level). Use `KFold(n_splits=5, shuffle=True, random_state=42)`. Articles get fully assigned to a fold; pairs from one article never split across folds.
2. For each fold:
   - Train on 4 folds, predict on the held-out fold
   - Threshold at 0.5 (or sweep thresholds on a training-internal split if you want — keep this simple for now)
3. **Pool predictions across all folds**, then compute precision, recall, F1 over the full set of directed edges. Do not average per-fold F1.
4. Run with **3 random seeds** (42, 123, 456) and report mean ± std.

## Output Format

Save a results JSON with this structure:

```json
{
  "model_name": "allenai/scibert_scivocab_uncased",
  "config": {},
  "per_seed_results": [
    {
      "seed": 42,
      "precision": 0.412,
      "recall": 0.587,
      "f1": 0.484,
      "tp": 187,
      "fp": 267,
      "fn": 132
    }
  ],
  "aggregated": {
    "precision_mean": 0.408,
    "precision_std": 0.012,
    "recall_mean": 0.591,
    "recall_std": 0.018,
    "f1_mean": 0.482,
    "f1_std": 0.009
  },
  "per_article_predictions": {
    "1409.0466": {
      "predicted_edges": [["S3.E1", "S3.E3"]],
      "ground_truth_edges": [["S3.E1", "S3.E3"]]
    }
  }
}
```

## Notebook Structure

Organize the notebook into clearly labeled sections:

1. **Setup** — install dependencies (`transformers`, `torch`, `scikit-learn`, `beautifulsoup4`, `lxml`), import packages, set seeds, check GPU
2. **Configuration** — single cell with all hyperparameters and paths at the top, easy to modify
3. **Data loading** — load JSON dataset, parse HTML for each article (downloading from ar5iv if needed), extract equations and prose
4. **Feature construction** — build pair-level training examples with text + scalar features
5. **Dataset class** — PyTorch Dataset wrapping the tokenizer
6. **Model definition** — encoder + classification head with scalar feature concatenation
7. **Training loop** — single function that takes train/val splits and returns a trained model + predictions
8. **Cross-validation orchestrator** — runs 5-fold CV across 3 seeds, collects pooled predictions
9. **Evaluation** — computes precision/recall/F1 pooled across all folds and articles
10. **Results saving** — writes results JSON and prints summary table
11. **Sanity checks** — at the bottom, print a few example predictions vs. ground truth for manual inspection

## Important Implementation Notes

- **HTML parsing**: ar5iv outputs contain `<math>` tags with `alttext` attributes. Use those for equation text. Equations are wrapped in `<math>` elements; equation labels are usually in `id` attributes like `S3.E1`. Use BeautifulSoup with the `lxml` parser.
- **Watch for the article that ar5iv mis-renders** (mentioned in §6 of the paper — one article was excluded). The notebook should detect articles where an equation ID in the adjacency list isn't found in the HTML and log a warning, then skip that article (matching the paper's protocol of evaluating on 63 articles).
- **Don't substitute zero-edge graphs for failed articles** — match the paper's protocol of excluding them entirely.
- **Free Colab session limits**: a full 3-seed × 5-fold run on SciBERT should fit in a single ~6-hour session. Add checkpoint saving after each fold so a disconnect doesn't kill the run.
- **Print incremental progress**: log per-fold metrics as they complete, not just final aggregated numbers, so I can see if training is going off the rails before it finishes.
- **Add a `DEBUG_MODE = True` flag** near the top that runs a single fold with 1 epoch to validate the pipeline end-to-end before committing to the full 3×5 run.

## Suggested Code Quality

- Add docstrings to all functions
- Type hints where it improves clarity
- A `Config` dataclass at the top so all hyperparameters are in one place
- No global state — pass everything explicitly
- One `main()` function at the bottom that runs the whole pipeline

## What to Skip

- Don't implement hyperparameter search — fixed config is fine.
- Don't implement focal loss or fancy class-balancing — just class weighting in cross-entropy.
- Don't build a fancy CLI — the notebook config cell is the interface.
- Don't add Weights & Biases or other experiment tracking — JSON output is sufficient.

## Final Deliverable

One `.ipynb` file, runnable top-to-bottom on Colab free tier after uploading the dataset. Include a brief markdown cell at the top explaining what the notebook does and the expected runtime (estimate 2–4 hours on T4 for SciBERT, 5-fold CV, 3 seeds).
