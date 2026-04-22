from __future__ import annotations

import numpy as np


def predict_positive_class_proba(model, X):
    """
    Return positive-class probability if available.
    Otherwise transform decision function with a logistic map.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))

    raise ValueError("Model must implement predict_proba or decision_function.")