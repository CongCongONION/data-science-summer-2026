# NLP Extension - Real Estate Text Analytics and Model Validation

**Post-internship independent extension (November 2026).** This directory is
NOT part of the IDX Exchange internship deliverable and was not built during
the internship. It extends the frozen `reproduction/` pipeline with the
unstructured-data components the original project did not have: NLP entity
extraction, text classification, semantic retrieval, and a formal model
validation write-up.

## What it does

Starting from the 5,641 analysis-ready Sold records of the reproduction, it
generates MLS-style remark text (`PublicRemarks`) with known-by-construction
gold labels, then builds and evaluates:

| Component | Script | Evaluation |
|---|---|---|
| Entity / signal extraction (rules + regex) | `entity_extraction.py` | Per-field exact match and P/R/F1 vs gold labels |
| Condition classification (TF-IDF + logistic regression) | `intent_classifier.py` | Random split AND template-family holdout |
| Semantic retrieval (MiniLM embeddings + FAISS vs TF-IDF) | `embedding_index.py` | Recall@5/@10, MRR@10, nDCG@10 over 16 predicate-defined queries |
| Pricing ablation (hedonic OLS +/- text features) | `ablation.py` | Out-of-time R2 / RMSE, gold vs extracted features |

## Headline results (all from committed `output/*.json`)

- Extraction recovers the structured entities essentially perfectly **on
  this synthetic text** (beds / baths / sqft / year / HOA exact match 1.00
  after a documented bug fix; boolean mention F1 1.00). Read as an upper
  bound - see the limitations.
- Condition classification: macro-F1 **1.00** on a random split, but only
  **0.47** when entire phrasing templates are held out of training. The gap
  between those two numbers is the most important result in this folder:
  it is the difference between memorizing templates and generalizing.
- Retrieval (16 queries): the TF-IDF baseline **beats** the embedding
  system on ranking quality (nDCG@10 0.9219 vs 0.7361; MRR@10 0.9583 vs
  0.8906). On formulaic text whose vocabulary the queries share, keyword
  matching wins and semantic smoothing only blurs the top ranks - the
  challenger is not adopted for this corpus. (Recall@10 is near-zero for
  both by construction: several predicates match hundreds of documents,
  so the ranking metrics are the informative ones.)
- Ablation: text features add **nothing** to the hedonic pricing model
  (out-of-time R2 0.639 baseline vs 0.639 with gold text features, 0.639
  with extracted features). The text channel's value is risk flagging
  (tenant occupancy, flood mention, condition triage), not price prediction.

## The honesty constraints (read before citing any number)

1. The underlying dataset is the reproduction's **synthetic sample data** in
   the CRMLS schema; the licensed CRMLS exports are not in any repository.
2. The remark text is **template-generated from the gold labels**, so every
   extraction / classification score is an upper bound relative to real MLS
   remarks. `validation_report.md` states this in its first section and the
   template-holdout experiment quantifies how much of the headline score is
   template memorization.
3. Nothing here is an IDX Exchange work product. Dates, provenance, and the
   internship/reproduction boundary are documented in `model_card.md`.

## Run it

```bash
pip install -r requirements.txt
python generate_text_data.py   # writes data/property_text.csv, data/gold_labels.csv
python entity_extraction.py    # writes output/extraction_metrics.json
python intent_classifier.py    # writes output/classification*_metrics.json
python embedding_index.py      # writes output/retrieval_metrics.json (downloads the MiniLM model)
python ablation.py             # writes output/ablation_metrics.json
pytest tests/ -q               # integrity checks over the committed artifacts
```
