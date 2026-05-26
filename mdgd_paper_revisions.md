# MDGD Paper Revisions: Three Targeted Additions

This document contains two specific additions/revisions for the MDGD paper, intended to strengthen the methodology section and correct an error in the results discussion.

Table1: \label{tab:derivation_correctness}
  \centering
  \begin{threeparttable}
    \begin{tabular}{p{5cm}|ccc}
      \toprule
      \hline
      \textbf{Algorithm} & \textbf{Precision} & \textbf{Recall} & $\mathbf{F_1}$ \\
      \midrule
      \hline
      Brute Force          & $\mathbf{48.52}$ & $\mathbf{36.53}$ & $\mathbf{41.68}$ \\
      Token Similarity     & $14.68$ & $23.61$ & $18.10$ \\
      Naive Bayes\tnote{1} & $13.07$ & $28.13$ & $17.85$ \\
      \hline
      GPT-5-mini            & $50.02 \pm 0.43$ & $\mathbf{79.73 \pm 1.37}$ & $61.47 \pm 0.73$ \\
      Gemini-3-Flash-Preview        & $54.43 \pm 2.12$ & $79.72 \pm 2.16$ & $\mathbf{64.69 \pm 2.16}$ \\
      DeepSeek-V4-Flash     & $\mathbf{57.23 \pm 1.25}$ & $69.49 \pm 1.37$ & $62.76 \pm 1.06$ \\
      Llama-3.1-8B-Instruct          & $36.52 \pm 1.93$ & $52.91 \pm 1.09$ & $43.20 \pm 1.70$ \\
      \hline
      \bottomrule
    \end{tabular}
    \begin{tablenotes}
      \small
      \item[1] Pooled metrics aggregating all predictions across the $5$
      cross-validation folds before computing metrics.
    \end{tablenotes}
  \end{threeparttable}
\end{table*}

Table 2:
\subsection{Proposed Fixes and Results} \label{subsec:dg_i_pf}
Based on the analysis in \S~\ref{subsec:dg_ir_ea}, the authors test specific improvements targeting the two identified failure modes. All fixes use the winning P1 prompt and are evaluated under the same protocol as Table~\ref{tab:derivation_correctness}: pooled metrics across $63$ articles and $3$ independent runs. Results are reported in Table~\ref{tab:fixes}.

\begin{table*}[htbp]
    \caption{Algorithm Performance Metrics ($\%$) for Potential Fixes}
    \label{tab:fixes}
    \centering
    \begin{tabular}{p{8cm}|ccc}
      \toprule
      \hline
      \textbf{Algorithm} & \textbf{Precision} & \textbf{Recall} & $\mathbf{F_1}$ \\
      \midrule
      \hline
      Combination - GPT-5-mini          & $45.43\pm0.95$ & $82.41\pm1.11$ & $58.56\pm0.56$ \\
      Combination - Gemini-3-Flash-Preview  & $52.35\pm0.93$ & $\mathbf{84.33}\pm0.70$ & $\mathbf{64.60}\pm0.91$ \\
      Combination - DeepSeek-V4-Flash   & $\mathbf{56.64}\pm0.79$ & $71.71\pm0.48$ & $63.29\pm0.31$ \\
      \hline
      Edge Limitation - GPT-5-mini      & $50.20\pm0.75$ & $\mathbf{76.54}\pm1.00$ & $60.63\pm0.84$ \\
      Edge Limitation - Gemini-3-Flash-Preview & $55.54\pm0.88$ & $73.72\pm0.66$ & $\mathbf{63.35}\pm0.80$ \\
      Edge Limitation - DeepSeek-V4-Flash & $\mathbf{58.07}\pm1.13$ & $65.78\pm1.38$ & $61.68\pm1.04$ \\
      \hline
      Comb.\ + Postprocess - GPT-5-mini & $51.34\pm0.49$ & $44.52\pm0.29$ & $47.69\pm0.37$ \\
      Comb.\ + Postprocess - Gemini-3-Flash-Preview & $55.83\pm1.04$ & $\mathbf{46.37}\pm0.15$ &
  $\mathbf{50.66}\pm0.51$ \\
      Comb.\ + Postprocess - DeepSeek-V4-Flash & $\mathbf{57.74}\pm1.33$ & $39.55\pm0.89$ & $46.93\pm0.41$ \\
      \hline
      $2$-Shot - GPT-5-mini             & $50.25\pm1.05$ & $75.80\pm1.36$ & $60.43\pm1.15$ \\
      $2$-Shot - Gemini-3-Flash-Preview & $58.03\pm1.37$ & $\mathbf{80.40}\pm0.48$ & $\mathbf{67.40}\pm1.09$ \\
      $2$-Shot - DeepSeek-V4-Flash      & $\mathbf{61.13}\pm0.33$ & $71.94\pm1.26$ & $66.09\pm0.68$ \\
      \hline
      \hline
      \bottomrule
  \end{tabular}
\end{table*}


## 1. Justification for Whole-Dataset Evaluation (no held-out test set)

### Suggested addition to §4 or §6.1

The MDGD is positioned as an **evaluation benchmark** for derivation graph extraction, not as a training corpus. This framing is central to understanding why metrics are reported over the full 63-article set rather than a held-out test split.

The LLM experiments are zero-shot and few-shot: no parameters are fit on the dataset, so every article in the MDGD is a held-out test instance by construction. There is no leakage to control for because there is no training signal being absorbed by the models. This protocol mirrors that of established LLM evaluation benchmarks such as BIG-Bench, MATH, and GSM8K, where the entire dataset serves as a test set and no train/test split is reported.

The single learned method in the paper, Naive Bayes (§5.1), uses 5-fold cross-validation precisely because it fits parameters from the data. Pooling predictions across the held-out folds yields metrics over all 63 articles without any article ever being used for both training and evaluation. The same article-level 5-fold cross-validation protocol applies to the fine-tuned encoder baseline (Appendix X), ensuring fair comparison with the LLM methods which see no training data at all.

### Addressing prompt selection contamination

A reviewer could reasonably ask whether prompt selection on a 20-article subset, followed by evaluation on the full 64-article set (which includes those 20 articles), constitutes a form of contamination. We address this concern directly:

1. **No parameters are fit during prompt selection.** The 20-article subset is a development split for instruction design, analogous to a validation set used to choose a model architecture, not a training set used to fit weights.
2. **The effect size is bounded.** Prompt variants P0–P7 produce F1 differences within a relatively narrow range on the development subset (Table 11), and the winning variant P1 was robust across all four models tested, suggesting the selection generalizes rather than overfitting to the subset.
3. **Robustness check.** As a sanity check, Stage 2 metrics computed on the 43-article complement (the articles not used for prompt selection) yield F1 scores within [X.X]% of the full-set numbers reported in Table 1, indicating no meaningful subset-specific advantage.

*[Action item: actually run this robustness check and fill in the numbers. If they're within 1-2 F1 points, the framing above holds; if they're substantially different, you need to address the gap explicitly.]*


## 2. Paired Statistical Significance Testing

### Suggested addition as new subsection §6.4 (or appendix subsection)

#### Protocol

All pairwise method comparisons in Tables 1 and 2 are evaluated for statistical significance using a paired bootstrap test over articles. Articles are the unit of analysis because they are the unit of variation in this task: per-article performance varies substantially with article length, mathematical density, and derivation chain structure, while edge-level resampling would underestimate this variance.

For each comparison between methods A and B:

1. Compute per-article F1 for each method across all 63 articles. For LLM methods with 3 independent runs, the per-article F1 is averaged across runs before bootstrapping.
2. Draw 10,000 bootstrap samples of 63 articles with replacement.
3. For each bootstrap sample, recompute the pooled F1 for both methods using the sampled article set and take the difference Δ = F1(A) − F1(B).
4. Report the mean difference, 95% bootstrap confidence interval, and two-sided p-value (the fraction of bootstrap samples where the sign of Δ disagrees with the observed mean direction, multiplied by 2).
5. Apply Bonferroni correction for the number of comparisons within each comparison family (zero-shot vs baselines, fixes vs zero-shot, few-shot vs zero-shot).

As a secondary check, McNemar's test is applied at the edge level for each method pair, treating each candidate edge as a binary classification decision. McNemar p-values are reported alongside bootstrap p-values where they diverge meaningfully.

#### Comparisons tested

The significance testing focuses on four claim families:

1. **LLMs vs analytical baselines.** Each of GPT-5-mini, Gemini-3-Flash-Preview, and DeepSeek-V4-Flash zero-shot is compared against Brute Force, Token Similarity, and Naive Bayes. *Expected result: all LLM-vs-baseline comparisons significant.*

2. **Best LLM vs second-best LLM.** Gemini-3-Flash-Preview vs DeepSeek-V4-Flash vs GPT-5-mini zero-shot, pairwise. *Expected result: differences within run-to-run noise; significance unlikely.*

3. **Fixes vs zero-shot baselines.** Each fix (Combination, Edge Limitation, Post-Processing, 2-Shot) is compared against the same model's zero-shot result. *Expected result: 2-Shot significant for Gemini and DeepSeek; most other fixes not significant or significantly worse.*

4. **Few-shot vs zero-shot per model.** The headline improvement claim. *Expected result: significant for Gemini and DeepSeek; not significant or marginally negative for GPT-5-mini.*

#### Suggested results table

| Comparison | ΔF1 (%) | 95% CI | p-value | Significant (α=0.05, Bonferroni)? |
|---|---|---|---|---|
| Gemini ZS vs Brute Force | +23.0 | [X.X, X.X] | <0.001 | Yes |
| Gemini ZS vs GPT-5-mini ZS | +3.2 | [X.X, X.X] | 0.XX | No/Yes |
| Gemini 2-Shot vs Gemini ZS | +2.7 | [X.X, X.X] | 0.XX | No/Yes |
| DeepSeek 2-Shot vs DeepSeek ZS | +3.3 | [X.X, X.X] | 0.XX | No/Yes |
| GPT-5-mini 2-Shot vs GPT-5-mini ZS | −1.0 | [X.X, X.X] | 0.XX | No |
| Gemini Combination vs Gemini ZS | −0.1 | [X.X, X.X] | 0.XX | No |
| ... | ... | ... | ... | ... |

*[Action item: fill in actual values after running the bootstrap procedure on saved per-article F1 scores.]*

#### Why this matters for the paper

Adding significance testing accomplishes three things: it (a) confirms which improvements in §6.3 are real versus within-noise, (b) hardens the headline claim about few-shot prompting, and (c) preempts a likely reviewer concern about overinterpreting small F1 differences across runs with standard deviations of 0.5–2.2 percentage points.

### Implementation note

The bootstrap procedure requires per-article F1 scores saved during evaluation, not just pooled metrics. The evaluation pipeline should be modified to save a per-article CSV with columns `article_id`, `method`, `seed`, `precision`, `recall`, `f1`, `tp`, `fp`, `fn`. The bootstrap then operates on this CSV directly.


## Action Items Summary

1. **Write the whole-dataset justification** into §4 or as a preamble to §6.1.
2. **Run the prompt-selection robustness check** on the 43-article complement and fill in the numbers.
3. **Implement the paired bootstrap procedure** over per-article F1 scores. Save per-article CSV during evaluation; write a separate analysis script that produces the significance table.
4. **Add §6.4 (Statistical Significance)** with the protocol description and results table.
5. **Correct the GPT-5-mini few-shot text** in §6.3.4 — replace the factually wrong "improves more modestly" claim.
6. **Sweep the paper** for any other claim that few-shot or any fix improves "all" models; correct to "two of three" or specific model names.
7. **Update §6.3.5 and §7 conclusion** to reflect the model-dependent nature of few-shot improvements.
