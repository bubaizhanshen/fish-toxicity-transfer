"""Read the fixed chemical partitions and permitted target-label selections."""

from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from scipy import sparse


def read_split(inputs, split, budget):
    inputs = Path(inputs)
    folder = inputs / f"split_{split}"
    records = pd.read_csv(folder / "records.csv.gz", low_memory=False)
    records = records.loc[records.tax_group.eq("fish")].copy()
    required = [
        "record_id",
        "chemical_index",
        "group",
        "structure_key",
        "split",
        "endpoint_class",
        "endpoint_key",
        "effect",
        "endpoint_type",
        "species",
        "duration_h",
        "value_for_model",
    ]
    if records[required].isna().any().any() or not records.record_id.is_unique:
        raise ValueError("Missing required fields or duplicate record IDs")
    if not np.isfinite(records[["duration_h", "value_for_model"]].to_numpy()).all():
        raise ValueError("Nonfinite durations or responses")
    for unit in ["group", "structure_key"]:
        if records.groupby(unit).split.nunique().max() != 1:
            raise ValueError(f"A {unit} crosses outer partitions")
    selection = pd.read_csv(folder / "LOW_SELECTION.csv")
    selection = selection.loc[selection.shot_rank.between(1, budget)]
    if not selection.record_id.is_unique:
        raise ValueError("Duplicate target selections")
    low = records.loc[records.record_id.isin(selection.record_id)].copy()
    low["shot_rank"] = low.record_id.map(selection.set_index("record_id").shot_rank)
    if (
        len(low) != len(selection)
        or low.empty
        or not low.split.eq("train").all()
        or not low.endpoint_class.eq("chronic").all()
        or low.duplicated(["endpoint_key", "group"]).any()
    ):
        raise ValueError("Invalid target training selection")
    high = records.loc[
        records.split.eq("train") & records.endpoint_class.eq("acute")
    ].copy()
    contexts = pd.read_csv(inputs / "EVALUATED_ENDPOINTS.csv")
    endpoints = set(contexts.loc[contexts.tax_group.eq("fish"), "endpoint_key"])
    test = records.loc[
        records.split.eq("test") & records.endpoint_key.isin(endpoints)
    ].copy()
    if high.empty or test.empty or set(test.endpoint_key) != endpoints:
        raise ValueError("Missing source rows or evaluation contexts")
    if set(test.group) & (set(low.group) | set(high.group)):
        raise ValueError("Held-out chemicals receive training supervision")
    fp = sparse.load_npz(inputs / "descriptors/chemical_ecfp_radius2_2048.npz").tocsr()
    indices = records.chemical_index.to_numpy()
    if (
        (indices != indices.astype(int)).any()
        or indices.min() < 0
        or indices.max() >= fp.shape[0]
    ):
        raise ValueError("chemical_index is outside the fingerprint matrix")
    return low, high, test, endpoints, fp


def load_benchmark(inputs, split, budget, species):
    low, high, test, endpoints, fp = read_split(inputs, split, budget)
    task = split * 6 + [5, 20, 100].index(budget) * 2 + int(species)
    folds = pd.read_csv(Path(inputs) / f"tasks/task_{task:02d}/INNER_FOLDS.csv")
    if not folds.record_id.is_unique or set(folds.record_id) != set(low.record_id):
        raise ValueError("Inner folds do not match the permitted target labels")
    low["inner_fold"] = low.record_id.map(folds.set_index("record_id").inner_fold)
    if (
        set(low.inner_fold) != {0, 1, 2}
        or low.groupby("group").inner_fold.nunique().max() != 1
    ):
        raise ValueError("Invalid chemical-group folds")
    return low, high, test, endpoints, fp.astype(np.float32)


def import_inputs(archive, destination):
    """Copy only model inputs from a local numerical archive, excluding reports."""
    source, destination = Path(archive) / "inputs", Path(destination)
    files = [
        Path("EVALUATED_ENDPOINTS.csv"),
        Path("descriptors/chemical_ecfp_radius2_2048.npz"),
    ]
    for split in range(6):
        files += [
            Path(f"split_{split}/{name}")
            for name in ["records.csv.gz", "LOW_SELECTION.csv"]
        ]
    files += [Path(f"tasks/task_{task:02d}/INNER_FOLDS.csv") for task in range(36)]
    if not all((source / file).is_file() for file in files):
        raise FileNotFoundError(
            "The numerical archive is missing required model inputs"
        )
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    for file in files:
        (destination / file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / file, destination / file)
