"""Select configurations using target-label folds, then evaluate held-out chemicals."""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from .data import load_benchmark
from .metrics import macro_mae
from .models import configuration_count, fit_model

COLUMNS = [
    "record_id",
    "group",
    "structure_key",
    "endpoint_key",
    "species",
    "value_for_model",
]


def run(
    inputs,
    output,
    family,
    split,
    budget,
    species=False,
    embeddings=None,
    checkpoint=None,
):
    low, high, test, endpoints, fp = load_benchmark(inputs, split, budget, species)
    embeddings = np.load(embeddings, mmap_mode="r") if embeddings else None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cats = ["effect", "endpoint_type"] + (["species"] if species else [])
    modes = ["low"] if family in {"rbf_svr_low", "tabpfn_low"} else ["low", "joint"]
    scores, history = [], []
    for mode in modes:
        for config in range(configuration_count(family)):
            predictions = []
            for fold in range(3):
                valid = low.loc[low.inner_fold.eq(fold)]
                held_out = set(valid.group)
                train = low.loc[~low.group.isin(held_out)]
                if mode == "joint":
                    train = pd.concat(
                        [train, high.loc[~high.group.isin(held_out)]], ignore_index=True
                    )
                if set(train.group) & held_out:
                    raise ValueError(
                        "An inner validation chemical receives supervision"
                    )
                pred, fitted, note = fit_model(
                    train, valid, family, config, fp, cats, embeddings, checkpoint
                )
                predictions.append(
                    valid[COLUMNS].assign(prediction=pred, inner_fold=fold)
                )
                history.append(dict(mode=mode, config=config, fold=fold, **note))
                del fitted
            oof = pd.concat(predictions, ignore_index=True)
            if len(oof) != len(low) or not oof.record_id.is_unique:
                raise ValueError("Incomplete out-of-fold predictions")
            oof.to_csv(output / f"{mode}_config{config}_oof.csv.gz", index=False)
            scores.append(
                dict(
                    family=family,
                    mode=mode,
                    config=config,
                    oof_mae=macro_mae(oof, endpoints),
                )
            )
        candidates = [row for row in scores if row["mode"] == mode]
        # The original tree grid resolves exact score ties by configuration name.
        names = (
            ["lgbm_original", "lgbm_flexible", "lgbm_regularized"]
            if family == "lgbm"
            else None
        )
        chosen = min(
            candidates,
            key=lambda row: (
                row["oof_mae"],
                names[row["config"]] if names else row["config"],
            ),
        )
        train = low if mode == "low" else pd.concat([low, high], ignore_index=True)
        pred, fitted, note = fit_model(
            train, test, family, chosen["config"], fp, cats, embeddings, checkpoint
        )
        test[COLUMNS].assign(prediction=pred, split=split).to_csv(
            output / f"{mode}_predictions.csv.gz", index=False
        )
        joblib.dump(
            dict(fitted, categorical_inputs=cats),
            output / f"{mode}_model.joblib",
            compress=3,
        )
        (output / f"{mode}_selected.json").write_text(
            json.dumps(chosen, indent=2) + "\n"
        )
        history.append(dict(mode=mode, config=chosen["config"], stage="final", **note))
        del fitted
    pd.DataFrame(scores).to_csv(output / "cv_scores.csv", index=False)
    settings = dict(
        family=family,
        split=split,
        budget=budget,
        species=species,
        checkpoint=str(checkpoint) if checkpoint else None,
    )
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (output / "fit_history.json").write_text(json.dumps(history, indent=2) + "\n")


def select_family(folders, mode, output):
    """Choose a family by OOF error for one split, budget and species setting."""
    candidates, conditions, coverage = [], [], []
    for folder in map(Path, folders):
        settings = json.loads((folder / "settings.json").read_text())
        conditions.append(
            tuple(settings[key] for key in ["split", "budget", "species"])
        )
        chosen = json.loads((folder / f"{mode}_selected.json").read_text())
        oof = pd.read_csv(folder / f"{mode}_config{chosen['config']}_oof.csv.gz")
        coverage.append(
            oof[COLUMNS + ["inner_fold"]]
            .sort_values("record_id")
            .reset_index(drop=True)
        )
        candidates.append((chosen["oof_mae"], chosen["family"], folder))
    if (
        not candidates
        or len(set(conditions)) != 1
        or not all(coverage[0].equals(x) for x in coverage[1:])
    ):
        raise ValueError(
            "Family selection requires identical labels, folds and settings"
        )
    score, family, folder = min(candidates)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pd.read_csv(folder / f"{mode}_predictions.csv.gz").to_csv(
        output / "predictions.csv.gz", index=False
    )
    (output / "selection.json").write_text(
        json.dumps(dict(family=family, mode=mode, oof_mae=score), indent=2) + "\n"
    )
