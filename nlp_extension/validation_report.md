# Model validation report - NLP extension

Scope: the four components in this folder (rule-based extraction, TF-IDF +
logistic condition classification, MiniLM + FAISS retrieval, hedonic
ablation). Validation follows the pattern of conceptual soundness,
outcomes analysis, and ongoing-monitoring design. All figures are
re-derived from the committed artifacts in `output/`.

## 1. Conceptual soundness - and the central limitation

The design is sound for its stated purpose: demonstrate and evaluate an
NLP pipeline over real-estate text with reproducible metrics. Its central
limitation is structural and is stated before any result: **the text is
template-generated from the gold labels**, so the evaluation measures
(a) whether the pipeline is implemented correctly and (b) how performance
degrades when phrasing generalizes beyond the training templates. It does
not measure accuracy on real MLS remarks. Every headline number below is
paired with the control that keeps it honest.

## 2. Outcomes analysis

### 2.1 Extraction (`output/extraction_metrics.json`)

Post-fix results over 5,641 records: bedrooms, bathrooms, living area,
year built, and HOA amount exact match 1.00; tenant / flood / price-cut /
pool / ADU mention F1 1.00 (supports 667 / 383 / 939 / 1,224 / 337);
solar 3-way accuracy 1.00 with leased recall 1.00; keyword condition
accuracy 0.9665, uniform across property subtypes (0.962-0.971 in
`output/extraction_by_subtype.csv`).

**Defects found by validation (the reason this section exists):**

1. *Bathroom undercount.* Before the fix, bathroom exact match was
   **0.7465**. Root cause: one layout template writes "3 full baths" and
   the regex required the number to sit directly before "ba/bath".
   1,430 records - exactly that template's share - failed. Fix: added a
   "N full baths" pattern. Post-fix exact match 1.00.
2. *Solar owned/leased logic error.* Accuracy 0.9954 before the fix: when
   an unrelated sentence contained the word "lease" (e.g., a tenant
   lease), owned-solar records were classified "none". Fix: decide
   "leased" only when lease language attaches to the solar sentence.
   Post-fix accuracy 1.00.

On real text these rules would score materially lower; the transferable
finding is the *failure mode* - phrasing variants of a single field are
the dominant extraction risk, and per-field (not aggregate) monitoring is
what catches them.

### 2.2 Condition classification (`output/classification_metrics.json`,
`output/classification_holdout_metrics.json`)

- Random stratified split (4,512 train / 1,129 test): accuracy 1.00,
  macro-F1 1.00. **Rejected as evidence of generalization**: every
  phrasing template appears on both sides of the split.
- Template-family holdout (train on phrasing templates 0-1 per class,
  test on held-out templates 2-3; 2,845 / 2,796): accuracy **0.6602**,
  macro-F1 **0.4657**. This is the operative estimate. The model leans on
  template-specific vocabulary ("investor special", "move-in ready") and
  transfers only partially to unseen phrasings ("turnkey condition",
  "attention investors"). A production version needs either real labelled
  remarks or paraphrase augmentation, and this holdout - not the random
  split - is the acceptance metric.

### 2.3 Semantic retrieval (`output/retrieval_metrics.json`)

16 natural-language queries, gold relevance defined by predicates over
the labels (relevant-set sizes range from a few dozen for compound
predicates to 1,224 for pool mentions).

| System | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| TF-IDF cosine (baseline) | 0.0179 | 0.0324 | 0.9583 | 0.9219 |
| MiniLM embeddings + FAISS (challenger) | 0.0145 | 0.0249 | 0.8906 | 0.7361 |

Two validation observations:

1. **Metric selection matters more than the headline.** Recall@k is
   near-zero for both systems by construction - with hundreds of
   relevant documents, no top-10 list can recall more than a few
   percent. The ranking-quality metrics (MRR, nDCG) are the informative
   ones here, and any report leading with the recall figures alone
   would misstate both systems.
2. **The challenger loses, and is not adopted.** The queries share exact
   vocabulary with the formulaic remark text ("tenant occupied",
   "flood insurance", "price reduced"), so keyword matching is nearly
   optimal and the embeddings' semantic smoothing only blurs the top
   ranks. This is the expected outcome on controlled-vocabulary text;
   on real remarks - with agent paraphrases, abbreviations, and typos -
   the comparison could reverse, which is precisely why the baseline is
   retained and the embedding system is shelved rather than deleted.
   Governance conclusion: keep TF-IDF for this corpus; re-run this
   exact comparison before any future adoption of embeddings.

### 2.4 Pricing ablation (`output/ablation_metrics.json`)

Out-of-time split identical to the published hedonic model (train 3,271 /
test 2,370). The ablation harness re-implements the baseline spec and
lands at R2 0.6386 / RMSE $250,316 versus the published 0.6443 / $248,330;
the small gap is the log-to-price retransformation, and all comparisons
below use the harness consistently.

| Specification | Test R2 | Test RMSE |
|---|---:|---:|
| Baseline (structure only) | 0.6386 | $250,316 |
| + gold text features (perfect NLP) | 0.6386 | $250,329 |
| + extracted text features (deployed pipeline) | 0.6388 | $250,245 |

Conclusion: **no incremental pricing signal from text**, even with perfect
labels. The drawn condition label is a function of property age, which the
baseline already includes; the remaining text features are amenities and
mentions that this pricing spec does not reward. The text channel should
be justified on risk-flagging value (tenant occupancy, flood mention,
condition triage for review prioritization), never on valuation lift.
Reporting a null ablation result is deliberate: it is the control that
keeps the extension's claims scoped to what the evidence supports.

## 3. Error analysis summary

- Extraction errors concentrate in *phrasing variants of one field*, not
  in any property subtype (subtype condition accuracy spread is under one
  point). Monitoring implication: per-field exact-match dashboards, not
  a single aggregate score.
- Classification errors concentrate in *unseen phrasings* (Section 2.2).
  Aggregate accuracy on a random split concealed this completely.
- Retrieval errors: see Section 2.3.

## 4. Limitations and conditions of use

1. Synthetic data and synthetic text throughout; no figure in this folder
   may be quoted as performance on real CRMLS data or real MLS remarks.
2. Mention labels record what the text says, not facts about a property;
   they are triage signals requiring human verification.
3. The extension is a post-internship independent project (November 2026)
   and is not an IDX Exchange work product.
4. Revalidation triggers and monitoring design are specified in
   `model_card.md`.

## 5. Validation opinion

The pipeline is implemented correctly and its evaluation harness is
genuinely informative - it caught two extraction defects and it separates
memorization from generalization for the classifier. The components are
approved for what they are: a documented, reproducible demonstration with
honest bounds. They are not approved, and must not be represented, as
validated performance on real-world listing text.
