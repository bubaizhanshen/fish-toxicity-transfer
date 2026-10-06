"""Regression families and the parameter grids used in model selection."""

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import LinearSVR, SVR
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from .features import feature_matrices
from .metrics import weights

SEED = 20261006

FAMILIES = [
    "ridge",
    "elastic_net",
    "linear_svr",
    "nystroem_ridge",
    "knn",
    "extra_trees",
    "xgboost",
    "catboost",
    "mlp",
    "rbf_svr_low",
]


def fit_one(train, query, family, config, fp, cats):
    if family == "mlp":
        import inspect

        if "sample_weight" not in inspect.signature(MLPRegressor.fit).parameters:
            raise RuntimeError(
                "MLP requires scikit-learn 1.7; install the mlp extra in its own environment"
            )
    start = time.time()
    a, b, prep = feature_matrices(train, query, fp, cats, family)
    y = train.value_for_model.to_numpy(float)
    w = weights(train)
    parameters = dict(family=family, config=config, seed=SEED)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if family == "lgbm":
            from lightgbm import LGBMRegressor

            settings = [
                dict(n_estimators=150, num_leaves=15, min_child_samples=2),
                dict(n_estimators=500, num_leaves=31, min_child_samples=5),
                dict(n_estimators=500, num_leaves=15, min_child_samples=10),
            ][config]
            model = LGBMRegressor(
                **settings,
                learning_rate=0.05,
                reg_lambda=1,
                n_jobs=1,
                random_state=SEED,
                verbosity=-1
            )
        elif family == "rf":
            settings = [
                dict(max_depth=None, min_samples_leaf=2, max_features="sqrt"),
                dict(max_depth=20, min_samples_leaf=5, max_features=0.1),
            ][config]
            model = RandomForestRegressor(
                n_estimators=200, **settings, n_jobs=1, random_state=SEED
            )
        elif family == "ridge":
            model = Ridge(
                alpha=[1.0, 10.0, 100.0][config],
                solver="lsqr",
                tol=0.0001,
                max_iter=10000,
            )
        elif family == "elastic_net":
            model = ElasticNet(
                alpha=[0.001, 0.01, 0.1][config],
                l1_ratio=0.5,
                max_iter=10000,
                tol=0.001,
                random_state=SEED,
            )
        elif family == "linear_svr":
            model = LinearSVR(
                C=[0.1, 1.0, 10.0][config],
                epsilon=0.1,
                loss="squared_epsilon_insensitive",
                dual=False,
                max_iter=10000,
                tol=0.0001,
                random_state=SEED,
            )
        elif family == "nystroem_ridge":
            gamma = [1.0, 1.0, 0.1][config] / a.shape[1]
            kernel = Nystroem(
                kernel="rbf",
                gamma=gamma,
                n_components=min(256, len(train)),
                random_state=SEED,
                n_jobs=1,
            )
            a = kernel.fit_transform(a)
            b = kernel.transform(b)
            prep["kernel"] = kernel
            model = Ridge(
                alpha=[1.0, 10.0, 1.0][config],
                solver="lsqr",
                tol=0.0001,
                max_iter=10000,
            )
        elif family == "knn":
            k = min([3, 10, 30][config], len(train))
            model = NearestNeighbors(
                n_neighbors=k, algorithm="brute", metric="euclidean", n_jobs=1
            )
            model.fit(a)
            distances, indices = model.kneighbors(b)
            neighbour_weights = w[indices] / np.maximum(distances, 1e-12)
            zeros = (
                distances.le(1e-12)
                if isinstance(distances, pd.DataFrame)
                else distances <= 1e-12
            )
            has_zero = zeros.any(axis=1)
            neighbour_weights[has_zero] = w[indices[has_zero]] * zeros[has_zero]
            prediction = (neighbour_weights * y[indices]).sum(
                axis=1
            ) / neighbour_weights.sum(axis=1)
            prep.update(training_features=a, training_targets=y, training_weights=w)
        elif family == "extra_trees":
            settings = [
                dict(max_depth=None, min_samples_leaf=2, max_features="sqrt"),
                dict(max_depth=20, min_samples_leaf=5, max_features=0.1),
                dict(max_depth=None, min_samples_leaf=5, max_features=0.5),
            ][config]
            model = ExtraTreesRegressor(
                n_estimators=200, random_state=SEED, n_jobs=1, **settings
            )
        elif family == "xgboost":
            from xgboost import XGBRegressor

            model = XGBRegressor(
                n_estimators=400,
                max_depth=[3, 6, 9][config],
                min_child_weight=[1, 5, 10][config],
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1,
                tree_method="hist",
                device="cpu",
                n_jobs=1,
                random_state=SEED,
                objective="reg:squarederror",
            )
        elif family == "catboost":
            from catboost import CatBoostRegressor

            model = CatBoostRegressor(
                iterations=500,
                depth=[4, 6, 8][config],
                l2_leaf_reg=[3, 5, 10][config],
                learning_rate=0.05,
                loss_function="RMSE",
                thread_count=1,
                random_seed=SEED,
                verbose=False,
                allow_writing_files=False,
            )
        elif family == "mlp":
            model = MLPRegressor(
                hidden_layer_sizes=[(64,), (128, 64), (128, 64)][config],
                alpha=[0.0001, 0.0001, 0.01][config],
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                batch_size=min(256, len(train)),
                max_iter=500,
                tol=0.0001,
                n_iter_no_change=30,
                early_stopping=False,
                random_state=SEED,
            )
        elif family == "rbf_svr_low":
            model = SVR(
                kernel="rbf",
                C=[1.0, 10.0, 100.0][config],
                gamma=["scale", "scale", 0.001][config],
                epsilon=0.1,
                cache_size=512,
            )
        else:
            raise ValueError(family)
        if family != "knn":
            model.fit(a, y, sample_weight=w)
            prediction = model.predict(b)
    elapsed = time.time() - start
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite model output")
    note = dict(
        seconds=elapsed,
        training_rows=len(train),
        query_rows=len(query),
        input_features=a.shape[1],
        n_iter=np.asarray(getattr(model, "n_iter_", [])).tolist(),
        warnings=[str(x.message) for x in captured],
    )
    if family == "mlp":
        note["training_loss_curve"] = model.loss_curve_
    try:
        parameters["estimator_parameters"] = model.get_params()
    except AttributeError:
        pass
    return (prediction, model, prep, note, parameters)


FAMILIES += ["lgbm", "rf", "chemberta_lgbm", "tabpfn_low"]


def configuration_count(family):
    return 2 if family in {"rf", "tabpfn_low"} else 3


def fit_model(train, query, family, config, fp, cats, embeddings=None, checkpoint=None):
    """Fit on training rows and predict query rows without accessing query labels."""
    if family == "chemberta_lgbm":
        if embeddings is None:
            raise ValueError(
                "Supply --embeddings for the frozen molecular representation"
            )
        from .representations import fit_head

        pred, model, encoder, note = fit_head(train, query, config, embeddings, cats)
        return pred, {"model": model, "encoder": encoder}, note
    if family == "tabpfn_low":
        if checkpoint is None:
            raise ValueError("Supply --checkpoint for the TabPFN v2 regressor")
        from .representations import fit_tabpfn

        pred, model, prep, note = fit_tabpfn(
            train, query, [4, 8][config], fp, cats, checkpoint
        )
        return pred, {"preprocessing": prep, "checkpoint": str(checkpoint)}, note
    pred, model, prep, note, params = fit_one(train, query, family, config, fp, cats)
    return pred, {"model": model, "preprocessing": prep, "parameters": params}, note
