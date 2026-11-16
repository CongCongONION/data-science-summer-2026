# Annotation guidelines - NLP extension gold labels

The gold labels in `data/gold_labels.csv` are **known by construction**: the
generator (`generate_text_data.py`) draws the mention labels first and then
writes the remark text to express them. There is no human annotation pass;
these guidelines document the label definitions so the evaluation is
auditable and reproducible.

## Labels

| Field | Values | Definition |
|---|---|---|
| `condition` | turnkey_updated / original_condition / fixer_upper | The dominant condition statement in the remarks: renovated/move-in-ready, original but maintained, or explicitly distressed / as-is / rehab-needed. Drawn conditional on property age (older stock is more likely original or fixer). |
| `tenant_occupied` | 0/1 | The remarks state the property is leased or tenant-occupied at sale. |
| `hoa_amount` | USD/month, 0 = none | The monthly HOA / association dues stated in the remarks. 0 means no HOA is mentioned (not that none exists in reality). |
| `flood_mention` | 0/1 | The remarks mention a flood zone / flood plain or a flood-insurance requirement. This records a *text mention only* - it is not a determination that the property carries flood risk. |
| `price_reduced_mention` | 0/1 | The remarks advertise a price reduction or a motivated seller. |
| `pool_mention` | 0/1 | A pool or spa is mentioned as a property feature. |
| `solar_mention` | owned / leased / none | Solar panels are mentioned, split by whether the system is owned (conveys) or leased (buyer assumes the lease - the risk-relevant case). |
| `adu_mention` | 0/1 | An accessory dwelling unit is mentioned. |

## Rules and edge cases

- A label is 1 only when the corresponding statement is *expressed in the
  text*. Absence of a mention is labelled 0 even where the underlying fact
  might be true (e.g., a condo with no HOA sentence still has `hoa_amount=0`
  if no dues were written into its remarks).
- `solar_mention=leased` requires lease language attached to the solar
  system, not merely the word "lease" elsewhere (a tenant lease does not
  make solar leased).
- `condition` is single-label: where a generated remark could read both
  ways, the generator's drawn class governs.
- Structured entities (bedrooms, bathrooms, living area, year built) are
  not in this file; their ground truth is the source dataset itself
  (`reproduction/output/sold_analysis_ready.csv`).

## Known limitation

Because labels generate the text, extraction and classification scores in
this folder are upper bounds relative to real MLS remarks. A production
version would replace this file with human-adjudicated labels on real,
licensed remarks and re-run the identical evaluation scripts.
