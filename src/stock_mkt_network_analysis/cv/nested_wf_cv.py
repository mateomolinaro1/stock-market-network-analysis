"""
Nested walk-forward cross-validation for time series classification with a rolling feature pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from stock_mkt_network_analysis.cv.folds import build_rolling_time_series_folds
from stock_mkt_network_analysis.cv.feature_pipeline import RollingNetworkFeaturePipeline
from stock_mkt_network_analysis.models.utils import predict_positive_class_proba
from stock_mkt_network_analysis.utils.utils import align_x_y


@dataclass
class NestedCVResult:
    predictions: pd.DataFrame
    selection_history: pd.DataFrame


@dataclass
class NestedWalkForwardCV:
    feature_pipeline: RollingNetworkFeaturePipeline
    model: Any
    threshold_grid: Sequence[float]
    hyperparameter_grid: Sequence[Dict[str, Any]]
    scoring_func: Callable
    inner_train_size: int
    inner_val_size: int
    inner_step_size: Optional[int] = 1
    min_outer_train_size: Optional[int] = None
    """
    Parameters:
    - min_outer_train_size: Minimum data required before starting predictions, if not set, it defaults to the sum of
        lookback, inner_train_size and inner_val_size.
    """

    def run(
        self,
        returns: pd.DataFrame,
        target: pd.Series,
        outer_test_dates: Sequence[pd.Timestamp],
    ) -> NestedCVResult:
        """
        At each test date:
        1. Use past data → outer_train
        2. Build inner folds
        3. For each (threshold, hyperparams):
            - train on inner_train
            - validate on inner_val
        4. Pick the best combo
        5. Refit on full outer_train
        6. Predict at test date

        :param returns:
        :param target:
        :param outer_test_dates:
        :return:
        """

        returns = returns.sort_index()
        target = target.sort_index()
        outer_test_dates = pd.Index(outer_test_dates).sort_values()

        if self.min_outer_train_size is None:
            self.min_outer_train_size = (
                self.feature_pipeline.correlation_estimator.lookback
                + self.inner_train_size
                + self.inner_val_size
            )

        prediction_records: List[Dict[str, Any]] = []
        selection_records: List[Dict[str, Any]] = []

        for t_test in outer_test_dates:
            if t_test not in returns.index or t_test not in target.index:
                continue

            outer_train_dates = returns.index[returns.index < t_test]
            if len(outer_train_dates) < self.min_outer_train_size:
                continue

            inner_folds = build_rolling_time_series_folds(
                dates=outer_train_dates,
                train_size=self.inner_train_size,
                val_size=self.inner_val_size,
                step_size=self.inner_step_size,
            )

            if len(inner_folds) == 0:
                continue

            best_score = -np.inf
            best_threshold = None
            best_theta = None

            for c in self.threshold_grid:
                for theta in self.hyperparameter_grid:
                    cv_scores = []

                    for inner_train_dates, inner_val_dates in inner_folds:
                        inner_train_returns = returns.loc[inner_train_dates]
                        inner_train_target = target.loc[inner_train_dates]

                        X_train = self.feature_pipeline.make_features(
                            returns=inner_train_returns,
                            threshold=c,
                        )
                        X_train, y_train = align_x_y(X_train, inner_train_target)
                        # Shift -horizon_prediction y_train because we want to predict the future
                        X_val = self.feature_pipeline.make_features_for_dates(
                            returns=returns.loc[returns.index <= inner_val_dates[-1]],
                            target_dates=inner_val_dates,
                            threshold=c,
                        )
                        y_val = target.loc[inner_val_dates]
                        X_val, y_val = align_x_y(X_val, y_val)

                        if len(X_train) == 0 or len(X_val) == 0:
                            continue

                        model = clone(self.model).set_params(**theta)
                        model.fit(X_train, y_train)

                        yhat_val = predict_positive_class_proba(model, X_val)
                        score = self.scoring_func(y_val.values, yhat_val)

                        if not np.isnan(score):
                            cv_scores.append(score)

                    mean_score = float(np.mean(cv_scores)) if len(cv_scores) > 0 else -np.inf

                    if mean_score > best_score:
                        best_score = mean_score
                        best_threshold = c
                        best_theta = theta

            if best_threshold is None or best_theta is None:
                continue

            X_outer_train = self.feature_pipeline.make_features(
                returns=returns.loc[outer_train_dates],
                threshold=best_threshold,
            )
            y_outer_train = target.loc[outer_train_dates]
            X_outer_train, y_outer_train = align_x_y(X_outer_train, y_outer_train)

            X_outer_test = self.feature_pipeline.make_features_for_dates(
                returns=returns.loc[returns.index <= t_test],
                target_dates=[t_test],
                threshold=best_threshold,
            )
            y_outer_test = target.loc[[t_test]]
            X_outer_test, y_outer_test = align_x_y(X_outer_test, y_outer_test)

            if len(X_outer_train) == 0 or len(X_outer_test) == 0:
                continue

            final_model = clone(self.model).set_params(**best_theta)
            final_model.fit(X_outer_train, y_outer_train)

            yhat_test = predict_positive_class_proba(final_model, X_outer_test)

            prediction_records.append(
                {
                    "date": t_test,
                    "y_true": int(y_outer_test.iloc[0]),
                    "y_pred": float(yhat_test[0]),
                    "best_threshold": float(best_threshold),
                    "inner_cv_score": float(best_score),
                }
            )

            selection_records.append(
                {
                    "date": t_test,
                    "best_threshold": float(best_threshold),
                    "best_theta": best_theta,
                    "inner_cv_score": float(best_score),
                    "n_outer_train_obs": int(len(X_outer_train)),
                }
            )

            self.feature_pipeline.evict_before(outer_train_dates[0])

        predictions = pd.DataFrame(prediction_records).set_index("date").sort_index()
        selection_history = pd.DataFrame(selection_records).set_index("date").sort_index()

        return NestedCVResult(
            predictions=predictions,
            selection_history=selection_history,
        )