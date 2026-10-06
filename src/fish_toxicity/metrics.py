"""Structure-balanced errors and paired chemical bootstrap intervals."""

import numpy as np
import pandas as pd


def macro_mae(z, endpoints):
    a = z[z.endpoint_key.isin(endpoints)].copy()
    a["ae"] = (a.prediction - a.value_for_model).abs()
    return float(
        a.groupby(["endpoint_key", "structure_key"])
        .ae.mean()
        .groupby("endpoint_key")
        .mean()
        .mean()
    )


def weights(train):
    counts = train.groupby(
        ["structure_key", "effect", "endpoint_type"]
    ).record_id.transform("size")
    w = 1 / np.sqrt(counts.to_numpy(float))
    return w / w.mean()


def error_metrics(frame):
    """A positive concentration error underestimates toxic potency."""
    error = frame.prediction - frame.value_for_model
    return frame.assign(
        mae=error.abs(),
        within10=error.abs().le(1).astype(float),
        underestimate=error.gt(1).astype(float),
        overestimate=error.lt(-1).astype(float),
    )


def summarize(frame):
    columns = ["mae", "within10", "underestimate", "overestimate"]
    return (
        error_metrics(frame)
        .groupby(["split", "endpoint_key", "structure_key"])[columns]
        .mean()
        .groupby(["split", "endpoint_key"])
        .mean()
        .groupby("split")
        .mean()
        .mean()
    )


def paired_bootstrap(candidate, baseline, draws=2000, seed=20261006):
    """Candidate minus baseline; jointly resample structures across all partitions.

    Models and label selections are held fixed. A negative MAE difference favors
    the candidate. Each context within each partition receives equal weight.
    """
    keys = ["split", "record_id"]
    identity = keys + ["structure_key", "endpoint_key", "species", "value_for_model"]
    a, b = [x.sort_values(keys).reset_index(drop=True) for x in (candidate, baseline)]
    if (
        a.duplicated(keys).any()
        or b.duplicated(keys).any()
        or not a[identity].equals(b[identity])
    ):
        raise ValueError("Predictions must cover the same records and responses")
    if not np.isfinite(a.prediction).all() or not np.isfinite(b.prediction).all():
        raise ValueError("Predictions must be finite")
    columns = ["mae", "within10", "underestimate", "overestimate"]
    diff = a[identity].copy()
    diff[columns] = error_metrics(a)[columns] - error_metrics(b)[columns]
    grouped = (
        diff.groupby(["split", "endpoint_key", "structure_key"])[columns]
        .mean()
        .reset_index()
    )
    structures = sorted(grouped.structure_key.unique())
    positions = {key: i for i, key in enumerate(structures)}
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(structures), np.full(len(structures), 1 / len(structures)), size=draws
    )
    partition_samples, partition_points = [], []
    for _, partition in grouped.groupby("split"):
        context_samples, context_points = [], []
        for _, context in partition.groupby("endpoint_key"):
            w = counts[:, context.structure_key.map(positions).to_numpy()]
            denominator = w.sum(axis=1)
            if (denominator == 0).any():
                raise ValueError(
                    "A resampled context is empty; intervals require broader coverage"
                )
            context_samples.append(
                w @ context[columns].to_numpy() / denominator[:, None]
            )
            context_points.append(context[columns].mean().to_numpy())
        partition_samples.append(np.mean(context_samples, axis=0))
        partition_points.append(np.mean(context_points, axis=0))
    samples = np.mean(partition_samples, axis=0)
    return pd.DataFrame(
        {
            "metric": columns,
            "difference": np.mean(partition_points, axis=0),
            "lower": np.quantile(samples, 0.025, axis=0),
            "upper": np.quantile(samples, 0.975, axis=0),
        }
    )
