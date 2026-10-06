"""Optional frozen molecular embeddings and TabPFN v2 comparator."""

import time
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from .metrics import weights as row_weights

SEED = 20261006


def fit_head(train, query, config, features, cats):
    from lightgbm import LGBMRegressor

    start = time.time()
    enc = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    enc.fit(train[cats].fillna("unknown"))

    def make(z):
        molecular = features[z.chemical_index.to_numpy(int)]
        if not np.isfinite(molecular).all():
            raise RuntimeError("Incomplete molecular representation")
        return sparse.hstack(
            [
                sparse.csr_matrix(molecular),
                enc.transform(z[cats].fillna("unknown")),
                sparse.csr_matrix(
                    np.log10(z.duration_h.clip(lower=1).to_numpy())
                    .astype(np.float32)
                    .reshape(-1, 1)
                ),
            ],
            format="csr",
        )

    a, b = (make(train), make(query))
    settings = [
        dict(n_estimators=150, num_leaves=15, min_child_samples=2),
        dict(n_estimators=500, num_leaves=31, min_child_samples=5),
        dict(n_estimators=500, num_leaves=15, min_child_samples=10),
    ][config]
    model = LGBMRegressor(
        learning_rate=0.05,
        reg_lambda=1.0,
        n_jobs=1,
        random_state=SEED,
        verbosity=-1,
        **settings
    )
    model.fit(
        a, train.value_for_model.to_numpy(float), sample_weight=row_weights(train)
    )
    pred = model.predict(b)
    if not np.isfinite(pred).all():
        raise RuntimeError("Nonfinite conditional tree output")
    return (
        pred,
        model,
        enc,
        dict(
            seconds=time.time() - start,
            training_rows=len(train),
            query_rows=len(query),
            input_features=a.shape[1],
        ),
    )


def fit_tabpfn(train, query, estimators, fp, cats, checkpoint):
    from tabpfn import TabPFNRegressor
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    ids = np.unique(train.chemical_index.to_numpy(int))
    svd = TruncatedSVD(
        n_components=min(128, len(ids) - 1, fp.shape[1] - 1),
        n_iter=5,
        random_state=SEED,
    )
    svd.fit(fp[ids])
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    enc.fit(train[cats].fillna("unknown"))

    def features(z):
        return np.column_stack(
            [
                svd.transform(fp[z.chemical_index.to_numpy(int)]),
                enc.transform(z[cats].fillna("unknown")),
                np.log10(z.duration_h.clip(lower=1).to_numpy()),
            ]
        ).astype(np.float32)

    x = features(train)
    qx = features(query)
    if x.shape[1] > 500:
        raise RuntimeError("Context feature limit exceeded; no silent truncation")
    cat_indices = list(range(svd.n_components, x.shape[1] - 1))
    model = TabPFNRegressor(
        n_estimators=estimators,
        device="cpu",
        n_jobs=1,
        model_path=checkpoint,
        categorical_features_indices=cat_indices,
        random_state=SEED,
        fit_mode="fit_preprocessors",
        inference_config={"MAX_UNIQUE_FOR_CATEGORICAL_FEATURES": 512},
        ignore_pretraining_limits=False,
    )
    then = time.time()
    model.fit(x, train.value_for_model.to_numpy(float))
    pred = np.concatenate(
        [
            model.predict(qx[i : i + 256], output_type="mean")
            for i in range(0, len(qx), 256)
        ]
    )
    if not np.isfinite(pred).all():
        raise RuntimeError("Nonfinite TabPFN output")
    note = dict(
        seconds=time.time() - then,
        training_rows=len(train),
        query_rows=len(query),
        input_features=x.shape[1],
        n_estimators=estimators,
        uniform_context=True,
        query_batch=256,
    )
    return (
        pred,
        model,
        dict(
            svd=svd,
            encoder=enc,
            categories=cats,
            training_x=x,
            training_y=train.value_for_model.to_numpy(float),
        ),
        note,
    )


def encode(molecules, checkpoint, output, matrix_rows):
    """Write frozen CLS embeddings indexed by chemical_index (100-token input)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if Path(output).exists():
        raise FileExistsError(output)
    molecules = molecules[["chemical_index", "smiles"]].drop_duplicates()
    if molecules.chemical_index.duplicated().any() or molecules.smiles.isna().any():
        raise ValueError("Each chemical_index must have one SMILES string")
    ids = molecules.chemical_index.to_numpy(int)
    if len(ids) == 0 or ids.min() < 0 or ids.max() >= matrix_rows:
        raise ValueError("Invalid chemical indices or matrix size")
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to("cpu").eval()
    features = np.full(
        (matrix_rows, model.config.hidden_size), np.nan, dtype=np.float32
    )
    with torch.no_grad():
        for start in range(0, len(molecules), 64):
            part = molecules.iloc[start : start + 64]
            tokens = tokenizer(
                part.smiles.tolist(),
                truncation=True,
                padding="max_length",
                max_length=100,
                return_tensors="pt",
            )
            features[part.chemical_index.to_numpy(int)] = (
                model(**tokens).last_hidden_state[:, 0].float().numpy()
            )
    if not np.isfinite(features[ids]).all():
        raise ValueError("Nonfinite molecular embeddings")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.save(output, features)
