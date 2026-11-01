"""Growth-only random-forest and kernel-ridge comparators for the external set."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler


def predict_classical(training_features, training_labels, query_features, settings):
    """Fit the two comparators using one frozen selection of growth records."""
    x = np.asarray(training_features)
    y = np.asarray(training_labels)
    q = np.asarray(query_features)
    if x.ndim != 2 or q.ndim != 2 or x.shape[1] != q.shape[1] or y.shape != (len(x),):
        raise ValueError("Features and labels have incompatible shapes")
    if not all(np.isfinite(v).all() for v in (x, y, q)):
        raise ValueError("Features and labels must be finite")
    mean = float(y.mean())
    forest = RandomForestRegressor(
        n_estimators=200, max_features=1.0,
        min_samples_leaf=settings["target_rf"]["leaf"],
        random_state=settings["seed"], n_jobs=1,
    ).fit(x, y - mean)
    scaler = StandardScaler().fit(x)
    params = settings["target_krr"]
    ridge = KernelRidge(
        kernel="rbf", alpha=params["alpha"], gamma=params["gamma_factor"] / x.shape[1]
    ).fit(scaler.transform(x), y - mean)
    return forest.predict(q) + mean, ridge.predict(scaler.transform(q)) + mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, help="external_baselines/split_N directory")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    settings = json.loads((args.inputs / "settings.json").read_text())
    with np.load(args.inputs / "inputs.npz") as data:
        rf, krr = predict_classical(data["train_features"], data["train_labels"],
                                    data["query_features"], settings)
        result = pd.DataFrame({"row_index": data["query_row_index"],
                               "split": settings["split"], "target_rf": rf, "target_krr": krr})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
