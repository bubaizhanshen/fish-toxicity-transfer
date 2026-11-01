"""Refit the growth-only and joint LightGBM external comparators."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import sparse


def _model(config):
    parameters = {key: value for key, value in config.items() if key not in {"name", "family"}}
    if config.get("family") != "lgbm":
        raise ValueError("The saved configuration must describe LightGBM")
    return LGBMRegressor(**parameters, learning_rate=0.05, reg_lambda=1,
                         n_jobs=1, random_state=20261006, verbosity=-1)


def predict_joint(source_x, source_y, source_w, target_x, target_y, target_w, query_x, config):
    """Fit the saved target-only, natural-weight and balanced-weight models."""
    matrices = (source_x, target_x, query_x)
    if any(m.shape[1] != source_x.shape[1] for m in matrices):
        raise ValueError("Feature dimensions differ")
    if len(source_y) != source_x.shape[0] or len(target_y) != target_x.shape[0]:
        raise ValueError("Label lengths differ")
    if len(source_w) != len(source_y) or len(target_w) != len(target_y):
        raise ValueError("Weight lengths differ")
    if any(not np.isfinite(m.data).all() for m in matrices) or any(
            not np.isfinite(v).all() for v in (source_y, source_w, target_y, target_w)):
        raise ValueError("Inputs must be finite")
    if np.any(source_w <= 0) or np.any(target_w <= 0):
        raise ValueError("Weights must be positive")
    model = _model(config).fit(target_x, target_y, sample_weight=target_w / target_w.mean())
    result = {"target_lgbm": model.predict(query_x)}
    xy = sparse.vstack([source_x, target_x], format="csr")
    y = np.concatenate([source_y, target_y])
    for mode in ("natural", "balanced"):
        sw = source_w if mode == "natural" else source_w * target_w.sum() / source_w.sum()
        w = np.concatenate([sw, target_w])
        model = _model(config).fit(xy, y, sample_weight=w / w.mean())
        result["joint_" + mode] = model.predict(query_x)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, help="external_joint/split_N directory")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    settings = json.loads((args.inputs / "settings.json").read_text())
    matrices = [sparse.load_npz(args.inputs / (name + "_features.npz"))
                for name in ("source", "target", "query")]
    with np.load(args.inputs / "labels_weights.npz") as a:
        result = predict_joint(matrices[0], a["source_y"], a["source_w"],
                               matrices[1], a["target_y"], a["target_w"], matrices[2], settings["config"])
        output = pd.DataFrame({"row_index": a["query_row_index"], "split": settings["split"], **result})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
