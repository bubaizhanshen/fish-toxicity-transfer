"""Mortality QSAR fitting and chemical-level cross-fitting for growth calibration."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CATEGORIES = ["effect", "endpoint_type"]


@dataclass
class MortalityQSAR:
    """A fitted source model with its training-only preprocessing."""

    model: object
    encoder: OneHotEncoder
    scaler: StandardScaler
    properties: pd.DataFrame
    fingerprints: object
    descriptor_names: Sequence[str]

    def _features(self, records):
        numeric = np.column_stack([
            self.properties.loc[records.group, self.descriptor_names],
            np.log10(records.duration_h.clip(lower=1)),
        ])
        return sparse.hstack([
            self.fingerprints[records.chemical_index.to_numpy(int)],
            sparse.csr_matrix(self.scaler.transform(numeric)),
            self.encoder.transform(records[CATEGORIES].fillna("unknown")),
        ], format="csr", dtype=np.float64)

    @classmethod
    def fit(cls, records, properties, fingerprints, descriptor_names):
        """Fit the source settings used in the reported mortality model.

        ``properties`` is indexed by reported-structure ``group``. Fingerprint
        rows are indexed by ``chemical_index``. Descriptor order must match the
        supplied 13-property inventory; species is not an input.
        """
        from lightgbm import LGBMRegressor

        if records.empty or len(descriptor_names) != 13:
            raise ValueError("Source rows and the 13 molecular properties are required.")
        if not records.effect.eq("MOR").all():
            raise ValueError("The source effect must be mortality (MOR).")
        if not records.endpoint_type.isin(["LC50", "EC50"]).all():
            raise ValueError("Source endpoints must be mortality LC50 or EC50.")
        if "tax_group" in records and not records.tax_group.eq("fish").all():
            raise ValueError("Source records must refer to fish.")
        if "split" in records and not records.split.eq("train").all():
            raise ValueError("Source fitting requires training records only.")
        if not properties.index.is_unique:
            raise ValueError("Each reported structure must have one descriptor row.")
        names = list(descriptor_names)
        numeric = np.column_stack([
            properties.loc[records.group, names],
            np.log10(records.duration_h.clip(lower=1)),
        ])
        if not np.isfinite(numeric).all():
            raise ValueError("Source descriptors and durations must be finite.")
        encoder = OneHotEncoder(handle_unknown="ignore").fit(
            records[CATEGORIES].fillna("unknown")
        )
        scaler = StandardScaler().fit(numeric)
        model = LGBMRegressor(
            n_estimators=500, num_leaves=15, min_child_samples=10,
            learning_rate=0.05, reg_lambda=1, n_jobs=1,
            random_state=20261006, verbosity=-1,
        )
        fitted = cls(model, encoder, scaler, properties, fingerprints.tocsr(), names)
        counts = records.groupby(
            ["structure_key", "effect", "endpoint_type"]
        ).record_id.transform("size").to_numpy()
        weights = 1 / np.sqrt(counts)
        weights /= weights.mean()
        model.fit(
            fitted._features(records), records.value_for_model,
            sample_weight=weights,
        )
        return fitted

    def predict(self, records):
        """Predict log10(mg/L) at the common mortality/LC50/96 h setting."""
        query = records.copy()
        query["effect"] = "MOR"
        query["endpoint_type"] = "LC50"
        query["duration_h"] = 96.0
        return np.asarray(self.model.predict(self._features(query)))


def cross_fitted_mortality(
    source, growth, query, properties, fingerprints, descriptor_names,
    split, excluded_parents=(),
):
    """Return source predictions for growth calibration and held-out chemicals.

    Pass the full permitted growth-training pool before applying label budgets.
    ``parent_group`` must be the standardized chemical identity used for all
    aliases. ``excluded_parents`` includes additional held-out candidates, such
    as reports that did not yield an eligible point NOEC. Inputs must already
    satisfy the study's endpoint recovery and outer-partition rules.

    Returns ``(growth_predictions, query_predictions, outer_model)`` in input
    row order. Growth predictions are out of their chemical's source fold.
    """
    if split not in range(6):
        raise ValueError("Partition must be between 0 and 5.")
    if not growth.effect.eq("GRO").all() or not growth.endpoint_type.eq("NOEC").all():
        raise ValueError("Growth training requires explicitly recovered GRO/NOEC.")
    if "split" in growth and not growth.split.eq("train").all():
        raise ValueError("Growth calibration requires training records only.")
    held = set(query.parent_group) | set(excluded_parents)
    if (set(source.parent_group) | set(growth.parent_group)) & held:
        raise ValueError("Held-out identities must be removed from both training pools.")
    parents = np.array(sorted(growth.parent_group.unique()))
    if len(parents) < 3:
        raise ValueError("At least three growth-training chemicals are required.")
    np.random.default_rng(2026101493 + split).shuffle(parents)
    assignment = {parent: i % 3 for i, parent in enumerate(parents)}
    folds = growth.parent_group.map(assignment)
    predictions = np.full(len(growth), np.nan)
    for fold in range(3):
        selected = folds.eq(fold).to_numpy()
        excluded = set(growth.loc[selected, "parent_group"])
        allowed = source.loc[~source.parent_group.isin(excluded)]
        fitted = MortalityQSAR.fit(allowed, properties, fingerprints, descriptor_names)
        predictions[selected] = fitted.predict(growth.loc[selected])
    if not np.isfinite(predictions).all():
        raise ValueError("Cross-fitted source predictions are incomplete.")
    outer = MortalityQSAR.fit(source, properties, fingerprints, descriptor_names)
    return predictions, outer.predict(query), outer
