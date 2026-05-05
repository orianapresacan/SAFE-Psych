# Experiments (Section 4 from the paper)

This folder contains the scripts used to compute the evaluation metrics, tables, and figures reported in the paper.

Before running the evaluation scripts, make sure the repository contains:

- `runs/`: with the model outputs produced during inference.
- `evaluations/data_gt.json`: the ground-truth annotations generated during data preprocessing.


## 4.1 Abstention Rate Analysis

Here, we evaluate whether models abstain appropriately when diagnostic information is insufficient and whether they avoid unnecessary abstention when diagnostic information is sufficient. *Full Information - Abstention Aware** and **Sequential Information - Abstention Aware** strategies.

Under- and over-abstention are computed as:
- **Under-abstention** = diagnosed insufficient cases / all insufficient cases
- **Over-abstention** = abstained sufficient cases / all sufficient cases

Run:
```bash
python evaluations/abstention_rate/process.py
python evaluations/abstention_rate/figure3.py
python evaluations/abstention_rate/appendixfigure3.py
```
- `figure3.py` generates the scatter plot for Figure 3, showing each model by under-abstention and over-abstention.
- `appendixfigure3.py` generates Appendix Figure 3, showing abstention rates separately for insufficient- and sufficient-information cases.

## 4.2 Stage #1 Clarification Analysis

This analysis evaluates the Sequential Information - No Abstention setting and measures whether models request clarification at Stage 1 on cases where the expert label also indicates Clarify.

- **Stage 1 clarification rate** = clarified at Stage 1 / all GT Stage-1-Clarify cases

Run:

```bash
python evaluations/clarification_seeking/process.py
python evaluations/clarification_seeking/figure4.py
```

## 4.3 Premature Diagnosis Analysis

This analysis focuses on the **Sequential Information** settings. For each case, it compares the step at which the model first produces a diagnosis to the expert-labeled earliest section where the available information becomes sufficient for diagnosis.

Diagnosis timing is classified as:
- **Premature**: first diagnosis occurs before the earliest sufficient section
- **On-time**: first diagnosis occurs at the earliest sufficient section
- **Late**: first diagnosis occurs after the earliest sufficient section
- **Never**: the model never produces a diagnosis

Run:

```bash
python evaluations/premature_diagnoses/process.py
python evaluations/premature_diagnoses/figures_5_6.py
```

The plotting script generates Figures 5 and 6 from the paper.

Figure 5 reports the percentage of cases in each timing category (**premature**, **on-time**, **late**, **never**) for each strategy.

Figure 6 shows cumulative diagnosis timing across stages. For each stage, it reports the percentage of cases already diagnosed by the model under the No Abstention and Abstention Aware settings. It also includes an expert sufficiency curve, showing when cases become sufficient for diagnosis according to expert labels.

## 4.4 Diagnosis Accuracy Analysis

Accuracy is computed at three levels:

- **Full ICD-10 code** (e.g., F32.1)
- **3-character ICD-10 category** (e.g., F32)
- **High-level ICD-10 diagnostic group** (e.g., F3)

Run:
```bash
python evaluations/diagnosis_accuracy/process.py
python evaluations/diagnosis_accuracy/figure8.py
python evaluations/diagnosis_accuracy/figure7.py
python evaluations/diagnosis_accuracy/figure9.py
```

Figure 8 computes 3-character ICD accuracy for each model under four settings (full information/no abstention, full information/abstention-aware, sequential/no abstention, and sequential/abstention-aware) and plots them as grouped bars.

Figure 7 compares diagnosis accuracy for premature versus on-time diagnoses under the Sequential Information - Abstention Aware setting, showing whether waiting until the expert-labeled sufficient section improves diagnostic correctness.

Figure 9 compares diagnosis accuracy between cases with full psychiatrist agreement and cases with any psychiatrist disagreement. A case is counted as full agreement only when annotators agreed on both diagnosis sufficiency and diagnosis label; otherwise, it is counted as disagreement.

Run:
```bash
python evaluations/diagnosis_accuracy/appendixfigure5.py
```

Appendix Figure 5 shows model accuracy by ground-truth ICD category under the **Full Information - No Abstention** setting. Diagnoses are grouped at the 3-character ICD level. The plot reports the top 6 and bottom 6 ICD categories by accuracy.

## 4.5 Reasoning Analysis

We evaluate Gemini 2.5 Flash with thinking disabled and with two inference-time thinking budgets: 128 and 2048 tokens.

Run:

```bash
python evaluations/reasoning_analysis/appendixfigure6.py
```

The script combines diagnosis accuracy and abstention metrics under the Sequential Information - Abstention Aware setting. It generates Appendix Figure 6, which compares 3-character ICD accuracy, under-abstention, and over-abstention across thinking budgets.

## 4.6 Seed Variability Analysis

This analysis estimates variability across repeated runs for a subset of models under the Full Information - Abstention Aware setting.

Run:
```bash
python evaluations/multiple_seed_runs/process.py
```
The script computes each metric separately for five seeds: 123, 42, 456, 789, and 2024. It reports mean, sample standard deviation, and 95% Student-t confidence intervals across seeds for under-abstention, over-abstention, and diagnosis accuracy.
