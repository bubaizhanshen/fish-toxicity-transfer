# Fish growth QSAR

Core analysis code for **Transfer Learning from Mortality QSARs Improves Fish
Growth Toxicity Prediction**.

The method fits a fish mortality QSAR, obtains chemical-level cross-fitted
predictions for growth-training compounds, and estimates an intercept from
growth no-observed-effect concentrations (NOECs). Each training chemical receives
equal total weight. Prediction for an untested chemical uses its mortality QSAR
estimate plus this intercept. Concentrations are expressed as log10(mg/L).

## Install

```bash
python -m pip install -e .
```

Python 3.10 or later is required. To fit the mortality model, also install:

```bash
python -m pip install -e '.[source]'
```

## Reproduce growth calibration

The prepared inputs are in `model_inputs` in Supporting Data S1. Each
dataset/partition directory contains `TRAIN.csv.gz`, `QUERY_METADATA.csv.gz`,
`EXACT_INPUTS.npz`, and `CHOICES.csv.gz`. The arrays contain chemically excluded
source predictions and preserve full-precision values. `benchmark`, `fels`, and
`japan` refer to the broad, historical, and Japanese evaluations.

```bash
fish-growth --inputs data/model_inputs/benchmark_split_0 \
  --split 0 --budget 20 --repeat 0 --output results/predictions.csv
```

Use `--regime one_record` for the frozen selection of one growth record per
chemical at budgets 5 or 20. The all-record regime supports budgets 5, 20, 80,
and `ALL`. Restricted budgets have ten selections (`--repeat 0` through `9`);
`ALL` uses repeat 0. Apply the command separately to each of six partitions and
three datasets.

The command writes predictions and a JSON summary of the fitted intercept and
training counts. It reads cached source predictions, so it does not require a
LightGBM refit or a foundation-model checkpoint.

## Fit the mortality source model

`source.cross_fitted_mortality` implements source training and threefold chemical
cross-fitting. It accepts curated source, growth-training, and query records,
a molecular-property table, and a sparse Morgan fingerprint matrix. Molecular
features are prepared before calling it.

```python
from fish_growth_qsar import InterceptCalibration
from fish_growth_qsar.source import cross_fitted_mortality

m_train, m_query, source_model = cross_fitted_mortality(
    source_records, growth_records, query_records,
    properties, fingerprints, descriptor_names,
    split=0, excluded_parents=held_out_candidates,
)
calibration = InterceptCalibration.fit(
    growth_records.value_for_model, m_train, growth_records.parent_group
)
predictions = calibration.predict_log_concentration(m_query)
```

Supply the full permitted growth-training pool for cross-fitting, then apply the
desired label selection when fitting `InterceptCalibration`. This preserves the
separation between source fitting and growth-data budgets.

The record schema uses `record_id`, `parent_group`, `group`, `chemical_index`,
`structure_key`, `effect`, `endpoint_type`, `duration_h`, and `value_for_model`.
Query records need only identifiers and feature indices. `parent_group` links
aliases of a standardized chemical; `group` indexes the reported structure in
the property table; `chemical_index` indexes fingerprint rows. Source records
are fish mortality LC50/EC50 observations. Growth labels must be explicitly
recovered GRO/NOEC records. Held-out candidates must be excluded from both
training pools before fitting.

The fingerprint is radius 2 with 2048 bits. The 13 properties, in the supplied
order, are parent logP, parent molecular weight, reported-structure molecular
weight, topological polar surface area, hydrogen-bond donors and acceptors,
ring count, aromatic-ring count, sp3-carbon fraction, rotatable bonds,
original-organic-component formal charge, heteroatoms, and heavy atoms.
Species is not a model input. Source rows retain their exposure durations;
queries use mortality LC50 at 96 h.

Observations originate from the [ECOTOX Knowledgebase](https://cfpub.epa.gov/ecotox/)
and the [TRIDENT release](https://doi.org/10.1126/sciadv.adk6669), with external
growth records described in the manuscript. Raw releases require endpoint,
identity, and unit harmonization. Use the supplied memberships and prepared
features for the reported analysis. Cached predictions provide the direct route
to reproducing the released calibration; numerical refits can vary with library
versions. No pretrained model checkpoint is distributed here.

## Evaluate predictions

`metrics.performance` calculates MAE, signed error, and large-error frequencies.
`metrics.comparisons` calculates paired differences using shared chemical
Bayesian-bootstrap weights. Input tables use `split`, `repeat`, `endpoint_key`,
`group`, `parent_group`, `record_id`, and observed `value_for_model`, plus one
prediction column per model.

```python
from fish_growth_qsar.metrics import performance, comparisons

scores = performance(predictions_table, models=["calibration", "direct"], seed=42)
contrasts = comparisons(
    predictions_table, primary="calibration", references=["direct"], seed=42
)
```

Run metrics separately for each dataset and training budget. Records are averaged
within chemical and endpoint group, with equal weighting of endpoint groups and
training contexts. The simultaneous limits in `comparisons` use the six-comparison
family specified in the paper; pointwise limits are also returned. Intervals
condition on fitted models and selections. Positive concentration error indicates
underestimated growth toxicity.

## Code layout

| Module | Purpose |
|---|---|
| `source.py` | Mortality LightGBM and chemical-level cross-fitting |
| `calibration.py` | Equal-chemical mean residual and calibrated prediction |
| `predict.py` | Calibration using released inputs and label selections |
| `metrics.py` | Original primary weighting, errors, and paired intervals |

This version targets growth NOEC. The earlier mixed-endpoint implementation
remains in Git history.
