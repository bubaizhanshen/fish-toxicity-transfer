# Fish toxicity transfer

Core code for evaluating whether high-effect fish mortality data improve prediction of low-effect concentrations when target labels are limited.

The comparison includes target-only training, joint training, and residual calibration of a source model. Chemical groups are held out together, including their source observations during inner validation. Configurations and model families are selected using out-of-fold target error.

## Install

```bash
python -m pip install -e .
# Optional model families:
python -m pip install -e '.[boosting,embeddings,mlp]'
# Run TabPFN in a separate environment:
python -m pip install -e '.[foundation]'
```

Python 3.10 or later is required. MLP training requires scikit-learn 1.7. TabPFN 2.0.9 requires scikit-learn below 1.7, so use a separate environment for that comparator and the v2 regressor checkpoint. Obtain pretrained checkpoints separately under their respective terms. Numerical results can vary with dependency versions.

## Inputs

This repository contains code only. The observations originate from [TRIDENT](https://doi.org/10.1126/sciadv.adk6669) and [ADORE](https://doi.org/10.1038/s41597-023-02612-2). Raw releases require harmonization; they cannot be passed directly to the training commands. The code consumes the study's prepared inputs, with this layout:

```text
data/
  EVALUATED_ENDPOINTS.csv
  descriptors/chemical_ecfp_radius2_2048.npz
  split_0/
    records.csv.gz
    LOW_SELECTION.csv
  ...
  split_5/
  tasks/task_00/INNER_FOLDS.csv
  ...
  tasks/task_35/INNER_FOLDS.csv
```

If you have the local numerical archive, copy its required inputs with:

```bash
fish-toxicity import-inputs /path/to/numerical_archive data
```

The processed inputs and model checkpoints are not distributed here. Exact numerical reproduction requires the original partitions, selections, descriptor matrix, and checkpoints; this refactoring has not been rerun as a full experimental benchmark.

**Record fields.** `record_id`, `tax_group`, `chemical_index`, `structure_key`, `group`, `split`, `endpoint_class`, `endpoint_key`, `effect`, `endpoint_type`, `species`, `duration_h`, and `value_for_model` are required. `chemical_index` indexes the fingerprint matrix (radius 2, 2,048 bits). `group` connects equivalent structures and shared chemical identifiers. `split` is `train`, `val`, or `test`; the outer validation partition is unused. Use the released log10(mg/L) response in `value_for_model`, not a recalculation from rounded linear concentrations.

**Effect levels.** The legacy input strings `acute` and `chronic` mean source high-effect and target low-effect records, respectively. They do not specify exposure duration. The target code pools NOEC and 0–10% effects; predictions are not uniformly EC10 values or protective concentrations. `endpoint_key` combines taxonomic group, biological effect, endpoint code, and duration bin. Legacy duration labels have boundaries at 24, 48, 96, 168, and 720 hours.

**Selections and folds.** `LOW_SELECTION.csv` contains `record_id` and `shot_rank`: budget K selects up to K chemical groups per target context, one record per group and context. `EVALUATED_ENDPOINTS.csv` contains `tax_group` and `endpoint_key`. Each fold file contains `record_id` and `inner_fold` (0, 1, or 2). Task numbering is `6 * split + 2 * budget_index + species`, where budget indices 0–2 mean K = 5, 20, 100 and species is 0 or 1.

## Train and compare

```bash
fish-toxicity benchmark --inputs data --output results/lgbm-s0-k5 \
  --family lgbm --split 0 --budget 5

fish-toxicity benchmark --inputs data --output results/rf-s0-k5 \
  --family rf --split 0 --budget 5

fish-toxicity select results/lgbm-s0-k5 results/rf-s0-k5 \
  --mode joint --output results/selected-s0-k5
```

Add `--species` to include species as an input. Run each family separately for the six partitions and three budgets. Selection is separate for target-only (`low`) and joint training (`joint`); pass all intended family folders with the same condition to `select`. Existing output directories are not overwritten. Each run writes out-of-fold scores, held-out predictions, the selected configuration, and fitted model components.

Available families:

- `lgbm`, `rf`, `ridge`, `elastic_net`, `linear_svr`, `nystroem_ridge`
- `knn`, `extra_trees`, `xgboost`, `catboost`, `mlp`, `chemberta_lgbm`
- `rbf_svr_low` and `tabpfn_low` (target-only comparators)

The molecular embedding model uses the frozen `seyonec/PubChem10M_SMILES_BPE_450k` encoder and CLS features, not fine-tuning. Supply an index-aligned `.npy` matrix through `--embeddings`, or create one from a CSV containing `chemical_index` and `smiles`:

```bash
N_ROWS=$(python -c "from scipy import sparse; print(sparse.load_npz('data/descriptors/chemical_ecfp_radius2_2048.npz').shape[0])")
fish-toxicity encode --molecules data/molecules.csv --checkpoint models/chemberta \
  --matrix-rows "$N_ROWS" --output data/embeddings.npy
```

The encoder uses the first token of the final hidden layer and a 100-token maximum. Use the study checkpoint for comparison with reported results. For `tabpfn_low`, pass `--checkpoint models/tabpfn-v2-regressor.ckpt`.

## Calibration and evaluation

```bash
fish-toxicity calibrate --inputs data --split 0 --output results/calibration-s0

fish-toxicity evaluate \
  --predictions results/lgbm-s0-k5/joint_predictions.csv.gz \
  --baseline results/lgbm-s0-k5/low_predictions.csv.gz \
  --output results/comparison-s0-k5
```

Calibration uses a fixed source LightGBM model and target residual medians, with global or context-specific corrections (shrinkage strength 5). It evaluates nested budgets of 1, 2, 5, 10, 20, 50, and 100. This control uses in-sample source predictions for calibration chemicals, not cross-fitted residuals.

Evaluation averages records within structure and context, then contexts and partitions equally. Metrics are log10-concentration MAE, the fraction within tenfold, and the fractions underestimating or overestimating potency by more than tenfold. A positive concentration error underestimates potency. Paired intervals report candidate minus baseline; a negative MAE difference favors the candidate. To evaluate all partitions, pass all six prediction files to each argument. The bootstrap resamples chemical structures jointly across partitions and conditions on fitted models and label selections.

## Code layout

- `data.py`: input loading and chemical-group separation
- `features.py`: preprocessing fitted on training rows
- `models.py`, `representations.py`: regression grids and optional pretrained models
- `benchmark.py`: inner validation, refitting, and family selection
- `calibration.py`: source-model residual corrections
- `metrics.py`: balanced metrics and paired intervals
- `cli.py`: command-line interface
