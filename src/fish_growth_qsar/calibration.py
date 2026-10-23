"""An intercept adjustment to fixed log10 concentration predictions.

The caller supplies chemical-held-out source predictions for calibration rows.
Source-model training, chemical exclusions and endpoint curation remain explicit
upstream responsibilities. This module does not train or fine-tune a backbone.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterceptCalibration:
    intercept: float
    n_chemicals: int
    n_records: int

    @classmethod
    def fit(cls, observed_log_concentration, source_prediction, chemical_ids):
        """Fit a constant by least squares, giving each chemical equal weight."""
        y = np.asarray(observed_log_concentration, dtype=float)
        m = np.asarray(source_prediction, dtype=float)
        groups = np.asarray(chemical_ids, dtype=str)
        if y.ndim != 1 or m.shape != y.shape or groups.shape != y.shape:
            raise ValueError("Supply aligned one-dimensional observations, predictions and identities.")
        if len(y) == 0 or not np.isfinite(y).all() or not np.isfinite(m).all():
            raise ValueError("Calibration requires finite, nonempty concentration data.")
        unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inverse]
        intercept = float(np.average(y - m, weights=weights))
        return cls(intercept, len(unique), len(y))

    def predict_log_concentration(self, source_prediction):
        """Return log10(mg/L) predictions, using the same scale as fitting."""
        return np.asarray(source_prediction, dtype=float) + self.intercept
