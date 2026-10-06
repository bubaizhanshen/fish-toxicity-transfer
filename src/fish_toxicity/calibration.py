"""Fixed source-model calibration using permitted target-label residuals."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from .data import read_split
from .metrics import weights
from .benchmark import COLUMNS


def residual_offsets(low, query_contexts, strength=5):
    """Shrink context residual medians toward the global residual median."""
    if strength <= 0 or low.empty or not np.isfinite(low.residual).all():
        raise ValueError(
            "Finite residuals and positive shrinkage strength are required"
        )
    global_offset = float(low.residual.median())
    context = low.groupby("endpoint_key").residual.agg(["median", "size"])
    offsets = {}
    for key in query_contexts:
        n = int(context.loc[key, "size"]) if key in context.index else 0
        local = float(context.loc[key, "median"]) if n else global_offset
        offsets[key] = (n * local + strength * global_offset) / (n + strength)
    return global_offset, offsets


def run(inputs, output, split, budgets=(1, 2, 5, 10, 20, 50, 100)):
    """Fit the fixed source model once; calibrate with each nested label budget.

    Residuals are not cross-fitted in this control: source training can include
    other effect-level records of the target calibration chemicals.
    """
    from lightgbm import LGBMRegressor

    low, high, test, endpoints, fp = read_split(inputs, split, max(budgets))
    low, high, test = [
        frame.sort_values("record_id").reset_index(drop=True)
        for frame in (low, high, test)
    ]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cats = ["effect", "endpoint_type"]
    encoder = OneHotEncoder(handle_unknown="ignore").fit(high[cats].fillna("unknown"))

    def features(frame):
        return sparse.hstack(
            [
                fp[frame.chemical_index.to_numpy(int)],
                encoder.transform(frame[cats].fillna("unknown")),
                sparse.csr_matrix(
                    np.log10(frame.duration_h.clip(lower=1).to_numpy()).reshape(-1, 1)
                ),
            ],
            format="csr",
        )

    model = LGBMRegressor(
        n_estimators=150,
        num_leaves=15,
        learning_rate=0.05,
        min_child_samples=2,
        reg_lambda=1,
        n_jobs=1,
        verbosity=-1,
        random_state=20261005,
    )
    model.fit(features(high), high.value_for_model, sample_weight=weights(high))
    source_prediction = model.predict(features(test))
    low["residual"] = low.value_for_model - model.predict(features(low))
    test[COLUMNS].assign(prediction=source_prediction, split=split).to_csv(
        output / "source_only.csv.gz", index=False
    )
    rows = []
    for budget in budgets:
        allowed = low.loc[low.shot_rank.le(budget)]
        global_offset, offsets = residual_offsets(allowed, endpoints)
        for method, adjustment in [
            ("global", global_offset),
            ("context", test.endpoint_key.map(offsets).to_numpy()),
        ]:
            test[COLUMNS].assign(
                prediction=source_prediction + adjustment, split=split
            ).to_csv(output / f"k{budget}_{method}.csv.gz", index=False)
        rows.extend(
            dict(
                budget=budget,
                endpoint_key=key,
                offset=value,
                global_offset=global_offset,
            )
            for key, value in offsets.items()
        )
    pd.DataFrame(rows).to_csv(output / "offsets.csv", index=False)
