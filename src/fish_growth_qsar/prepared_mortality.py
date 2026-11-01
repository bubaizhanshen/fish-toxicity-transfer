"""Refit the mortality source and cross-fitted growth calibration for external queries."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .joint import _model


FINGERPRINT_BITS = 2048
DESCRIPTORS = 13
STRUCTURAL_INDICATORS = 17
DURATION = FINGERPRINT_BITS + DESCRIPTORS + STRUCTURAL_INDICATORS


def _fit_source(source_x, source_y, structure_keys, query_x, config):
    """Fit the recorded source pipeline on one permitted chemical subset."""
    lc = source_x[:, DURATION + 4].toarray().ravel()
    ec = source_x[:, DURATION + 5].toarray().ravel()
    if not np.all((lc + ec) == 1):
        raise ValueError("Mortality source endpoints must be LC50 or EC50")
    cats = pd.DataFrame({"effect": "MOR", "endpoint_type": np.where(lc == 1, "LC50", "EC50")})
    # The original descriptor table produced column-contiguous numeric arrays.
    # Preserve that layout when fitting the scaler to reproduce its reductions.
    numeric = np.asfortranarray(np.column_stack([
        source_x[:, FINGERPRINT_BITS:FINGERPRINT_BITS + DESCRIPTORS].toarray(),
        np.maximum(source_x[:, DURATION].toarray().ravel(), 0),
    ]))
    scaler = StandardScaler().fit(numeric)
    encoder = OneHotEncoder(handle_unknown="ignore").fit(cats)
    x = sparse.hstack([source_x[:, :FINGERPRINT_BITS],
                       sparse.csr_matrix(scaler.transform(numeric)),
                       encoder.transform(cats)], format="csr", dtype=np.float64)
    groups = pd.DataFrame({"structure_key": structure_keys, "effect": cats.effect,
                           "endpoint_type": cats.endpoint_type})
    count = groups.groupby(["structure_key", "effect", "endpoint_type"])["structure_key"].transform("size").to_numpy()
    weights = 1 / np.sqrt(count)
    model = _model(config).fit(x, source_y, sample_weight=weights / weights.mean())
    qnumeric = np.column_stack([
        query_x[:, FINGERPRINT_BITS:FINGERPRINT_BITS + DESCRIPTORS].toarray(),
        np.full(query_x.shape[0], np.log10(96.0)),
    ])
    qcats = pd.DataFrame({"effect": ["MOR"] * query_x.shape[0],
                          "endpoint_type": ["LC50"] * query_x.shape[0]})
    q = sparse.hstack([query_x[:, :FINGERPRINT_BITS],
                       sparse.csr_matrix(scaler.transform(qnumeric)),
                       encoder.transform(qcats)], format="csr", dtype=np.float64)
    return np.asarray(model.predict(q))


def predict_source(source_x, source_y, source_keys, source_parent, target_x, target_y,
                   target_parent, target_fold, pool_parent, pool_fold, query_x, config):
    """Refit source folds, estimate the growth intercept, and predict queries."""
    if len(target_y) != target_x.shape[0] or len(source_y) != source_x.shape[0]:
        raise ValueError("Feature and label lengths differ")
    if len(set(target_parent)) != len(target_parent):
        raise ValueError("The supplied external selection must contain one record per chemical")
    assignment = dict(zip(pool_parent, pool_fold))
    if any(assignment.get(parent) != fold for parent, fold in zip(target_parent, target_fold)):
        raise ValueError("Selected growth chemicals do not match the source folds")
    target_oof = np.full(len(target_y), np.nan)
    for fold in range(3):
        selected = target_fold == fold
        if not selected.any():
            continue
        excluded = set(pool_parent[pool_fold == fold])
        allowed = ~np.isin(source_parent, list(excluded))
        target_oof[selected] = _fit_source(source_x[allowed], source_y[allowed], source_keys[allowed],
                                           target_x[selected], config)
    if not np.isfinite(target_oof).all():
        raise ValueError("Missing cross-fitted source predictions")
    source_query = _fit_source(source_x, source_y, source_keys, query_x, config)
    intercept = float(np.mean(target_y - target_oof))
    return target_oof, source_query, intercept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, help="external_joint/split_N directory")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    settings = json.loads((args.inputs / "settings.json").read_text())
    matrices = [sparse.load_npz(args.inputs / (name + "_features.npz"))
                for name in ("source", "target", "query")]
    with np.load(args.inputs / "labels_weights.npz") as a:
        oof, source, intercept = predict_source(
            matrices[0], a["source_y"], a["source_structure_key"], a["source_parent"],
            matrices[1], a["target_y"], a["target_parent"], a["target_fold"],
            a["growth_pool_parent"], a["growth_pool_fold"], matrices[2], settings["config"])
        target = pd.DataFrame({"record_id": a["target_record_id"], "mortality_oof": oof})
        query = pd.DataFrame({"row_index": a["query_row_index"], "split": settings["split"],
                              "mortality_source": source, "mortality_calibrated": source + intercept})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target.to_csv(args.output_dir / "growth_source_oof.csv", index=False)
    query.to_csv(args.output_dir / "external_predictions.csv", index=False)
    (args.output_dir / "intercept.json").write_text(json.dumps({"intercept": intercept}, indent=2) + "\n")


if __name__ == "__main__":
    main()
