# Model card - NLP extension components

## Provenance

- **What this is:** a post-internship independent extension (November 2026)
  of the IDX reproduction pipeline, built to add unstructured-text methods
  (extraction, classification, semantic retrieval) and a validation layer
  the original project did not have.
- **What this is not:** an IDX Exchange deliverable, a production system, or
  a model trained on real MLS remarks.
- **Data:** the reproduction's synthetic CRMLS-schema sample (5,641
  analysis-ready Sold records). Remark text is template-generated from
  structured fields and drawn mention labels (seed 20261115); gold labels
  are known by construction. See `annotation_guidelines.md`.

## Components

| Component | Method | Intended use | Out of scope |
|---|---|---|---|
| Entity extraction | Regex + keyword rules | Recover beds/baths/area/year/HOA and risk-relevant mentions (tenant, flood, solar lease) from remark text for downstream structuring | Free-form real-world remarks without re-validation |
| Condition classifier | TF-IDF (1-2 grams) + logistic regression | Triage listings by condition class for collateral review prioritization | Automated valuation or credit decisions |
| Semantic retrieval | all-MiniLM-L6-v2 embeddings, FAISS IndexFlatIP | Natural-language search over remark text where keyword search misses paraphrases | Replacing structured filters where a structured field exists |
| Hedonic ablation | OLS, published spec +/- text features | Test whether text adds pricing signal beyond structure | Price prediction |

## Evaluation summary

- Extraction: per-field results in `output/extraction_metrics.json`
  (post-fix: all exact-match rates 1.00, boolean F1 1.00, keyword condition
  accuracy 0.9665). One defect was found and fixed during validation
  (bathroom count missed the "N full baths" phrasing, exact match 0.7465
  before the fix; a solar owned/leased logic error, accuracy 0.9954 before
  the fix). Both are documented in `validation_report.md`.
- Classification: random-split macro-F1 1.00 (leaky by construction);
  template-family holdout macro-F1 0.4657, accuracy 0.6602 - the operative
  generalization estimate for this design.
- Retrieval: `output/retrieval_metrics.json` - TF-IDF baseline nDCG@10
  0.9219 / MRR@10 0.9583 vs MiniLM + FAISS 0.7361 / 0.8906 over 16
  predicate-defined queries. The embedding challenger does not beat the
  keyword baseline on this corpus and is not adopted (see validation
  report Section 2.3).
- Ablation: no incremental pricing signal from text features
  (R2 0.6386 baseline / 0.6386 gold / 0.6388 extracted).

## Limitations and risks

1. **Synthetic text ceiling.** All scores are upper bounds. Real remarks
   contain abbreviations, typos, agent shorthand, and statements that
   contradict the structured record; none of that is present here.
2. **Label-by-construction.** Gold labels generated the text, so evaluation
   measures pipeline correctness and phrasing generalization - not
   real-world extraction accuracy.
3. **Mentions are not facts.** A flood or tenant *mention* is a text signal
   to verify, never a determination about a property or a borrower. These
   outputs must not feed automated adverse decisions.
4. **Template holdout is a proxy.** Holding out phrasing families
   approximates, but does not equal, distribution shift to real text.
5. **Retrieval relevance is predicate-defined.** Gold relevance sets are
   derived from the labels, which favors neither system but does make the
   queries cleaner than real user queries.

## Monitoring / revalidation triggers (if this were productionized)

- Re-run the full evaluation suite on any template, model, or embedding
  change; fail the build if holdout macro-F1 drops more than 0.05.
- Replace synthetic text with licensed real remarks and human-adjudicated
  labels before any external claim; re-baseline all metrics at that point.
- Track extraction exact-match per field monthly; a drop in any single
  field (as happened with bathrooms) is the earliest drift signal.
