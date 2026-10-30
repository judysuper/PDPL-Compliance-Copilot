# Experiment E4 — Privacy Policy Clause Classifier

## Objective
Compare a TF-IDF + Logistic Regression baseline with a fine-tuned XLM-RoBERTa transformer for multi-label privacy-policy clause classification.

## Dataset
- Source: OPP-115
- Total policies used: 115
- Categories: 10
- Split strategy: policy-level multilabel stratified split
- Train policies: 79
- Validation policies: 18
- Test policies: 18
- Train paragraphs: 2619
- Validation paragraphs: 586
- Test paragraphs: 587

No policy appears in more than one split.

## Models
1. TF-IDF (word 1–2 grams) + One-vs-Rest Logistic Regression
2. Fine-tuned FacebookAI/xlm-roberta-base

Both models were evaluated on the same held-out policy-level test set.
Decision thresholds were tuned using validation data only.

## Overall Results

| Model | Macro-F1 | Micro-F1 | Hamming Loss | Subset Accuracy |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.7232 | 0.7508 | 0.0845 | 0.4310 |
| XLM-RoBERTa-base | 0.7679 | 0.7806 | 0.0779 | 0.4923 |

## Interpretation
XLM-RoBERTa achieved a higher Macro-F1 (0.7679) than TF-IDF + Logistic Regression (0.7232), an improvement of 0.0447.

The per-class results should be used to identify which privacy-practice categories benefit from contextual transformer representations and which remain difficult, especially low-frequency categories.

## Selected Model
**XLM-RoBERTa-base**

## Important Limitation
OPP-115 categories are not identical to Saudi PDPL legal requirements. The classifier output must therefore be mapped explicitly to a separate PDPL checklist before producing compliance-gap judgments.
