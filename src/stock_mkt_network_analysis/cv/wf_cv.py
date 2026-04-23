"""
Walk-forward cross-validation schemes for time series classification
with a rolling network feature pipeline.

Supported schemes:
- simple: rolling train / validation / test
- nested: outer test loop + inner rolling CV for model selection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
import logging

from stock_mkt_network_analysis.cv.folds import build_rolling_time_series_folds
from stock_mkt_network_analysis.cv.feature_pipeline import RollingNetworkFeaturePipeline
from stock_mkt_network_analysis.models.utils import predict_positive_class_proba
from stock_mkt_network_analysis.utils.utils import align_x_y


CVScheme = Literal["simple", "nested"]
logger = logging.getLogger(__name__)

@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    selection_history: pd.DataFrame


@dataclass
class SimpleRollingWalkForwardCV:
    feature_pipeline: RollingNetworkFeaturePipeline
    model: Any
    threshold_grid: Sequence[float]
    hyperparameter_grid: Sequence[Dict[str, Any]]
    scoring_func: Callable
    outer_train_size: int
    val_size: int
    min_required_history: Optional[int] = None

    def run(
        self,
        returns: pd.DataFrame,
        target: pd.Series,
        outer_test_dates: Sequence[pd.Timestamp],
    ) -> WalkForwardResult:

        returns = returns.sort_index()
        target = target.sort_index()
        outer_test_dates = pd.Index(outer_test_dates).sort_values()

        lookback = self.feature_pipeline.correlation_estimator.lookback

        if self.min_required_history is None:
            self.min_required_history = lookback + self.outer_train_size + self.val_size

        prediction_records: List[Dict[str, Any]] = []
        selection_records: List[Dict[str, Any]] = []

        all_dates = returns.index

        for i, t_test in enumerate(outer_test_dates):
            logger.info(f"Processing test date {t_test} ({i+1}/{len(outer_test_dates)})")
            if t_test not in returns.index or t_test not in target.index:
                continue

            test_loc = all_dates.get_loc(t_test)
            if isinstance(test_loc, slice):
                continue

            if test_loc < self.min_required_history:
                continue

            val_end = test_loc
            val_start = val_end - self.val_size

            train_end = val_start
            train_start = train_end - self.outer_train_size

            if train_start < 0:
                continue

            train_dates = all_dates[train_start:train_end]
            val_dates = all_dates[val_start:val_end]
            test_dates = pd.Index([t_test])

            best_score = -np.inf
            best_threshold = None
            best_theta = None

            for c in self.threshold_grid:
                logger.info(f"Evaluating threshold {c} for test date {t_test}")
                for theta in self.hyperparameter_grid:
                    logger.info(f"Evaluating hyperparameters {theta} for threshold {c} and test date {t_test}")
                    train_returns = returns.loc[train_dates]
                    train_target = target.loc[train_dates]

                    X_train = self.feature_pipeline.make_features(
                        returns=train_returns,
                        threshold=c,
                    )
                    X_train, y_train = align_x_y(X_train, train_target)

                    X_val = self.feature_pipeline.make_features_for_dates(
                        returns=returns.loc[returns.index <= val_dates[-1]],
                        target_dates=val_dates,
                        threshold=c,
                    )
                    y_val = target.loc[val_dates]
                    X_val, y_val = align_x_y(X_val, y_val)

                    if len(X_train) == 0 or len(X_val) == 0:
                        continue

                    candidate_model = clone(self.model).set_params(**theta)
                    candidate_model.fit(X_train, y_train)

                    yhat_val = predict_positive_class_proba(candidate_model, X_val)
                    score = self.scoring_func(y_val.values, yhat_val)

                    if not np.isnan(score) and score > best_score:
                        best_score = score
                        best_threshold = c
                        best_theta = theta

            if best_threshold is None or best_theta is None:
                continue

            refit_dates = all_dates[train_start:val_end]

            X_refit = self.feature_pipeline.make_features(
                returns=returns.loc[refit_dates],
                threshold=best_threshold,
            )
            y_refit = target.loc[refit_dates]
            X_refit, y_refit = align_x_y(X_refit, y_refit)

            X_test = self.feature_pipeline.make_features_for_dates(
                returns=returns.loc[returns.index <= t_test],
                target_dates=test_dates,
                threshold=best_threshold,
            )
            y_test = target.loc[test_dates]
            X_test, y_test = align_x_y(X_test, y_test)

            if len(X_refit) == 0 or len(X_test) == 0:
                continue

            final_model = clone(self.model).set_params(**best_theta)
            final_model.fit(X_refit, y_refit)

            yhat_test = predict_positive_class_proba(final_model, X_test)

            prediction_records.append(
                {
                    "date": t_test, # at date t_test w
                    "y_true": int(y_test.iloc[0]),
                    "y_pred": float(yhat_test[0]),
                    "best_threshold": float(best_threshold),
                    "validation_score": float(best_score),
                }
            )

            selection_records.append(
                {
                    "date": t_test, # at date t_test we store prediction of target variable for t+h
                    "scheme": "simple",
                    "train_start": train_dates[0],
                    "train_end": train_dates[-1],
                    "val_start": val_dates[0],
                    "val_end": val_dates[-1],
                    "best_threshold": float(best_threshold),
                    "best_theta": best_theta,
                    "validation_score": float(best_score),
                    "n_refit_obs": int(len(X_refit)),
                }
            )

            self.feature_pipeline.evict_before(train_dates[0])

        predictions = pd.DataFrame(prediction_records).set_index("date").sort_index()
        selection_history = pd.DataFrame(selection_records).set_index("date").sort_index()

        return WalkForwardResult(
            predictions=predictions,
            selection_history=selection_history,
        )


@dataclass
class NestedWalkForwardCV:
    feature_pipeline: RollingNetworkFeaturePipeline
    model: Any
    threshold_grid: Sequence[float]
    hyperparameter_grid: Sequence[Dict[str, Any]]
    scoring_func: Callable
    outer_train_size: int
    inner_train_size: int
    inner_val_size: int
    inner_step_size: Optional[int] = 1
    min_required_history: Optional[int] = None

    def run(
        self,
        returns: pd.DataFrame,
        target: pd.Series,
        outer_test_dates: Sequence[pd.Timestamp],
    ) -> WalkForwardResult:

        returns = returns.sort_index()
        target = target.sort_index()
        outer_test_dates = pd.Index(outer_test_dates).sort_values()

        lookback = self.feature_pipeline.correlation_estimator.lookback

        if self.min_required_history is None:
            self.min_required_history = max(
                lookback + self.outer_train_size,
                lookback + self.inner_train_size + self.inner_val_size,
            )

        prediction_records: List[Dict[str, Any]] = []
        selection_records: List[Dict[str, Any]] = []

        all_dates = returns.index

        cnt = 0
        for t_test in outer_test_dates:
            logger.info(f"Processing test date {t_test} ({cnt+1}/{len(outer_test_dates)})")
            cnt += 1
            if t_test not in returns.index or t_test not in target.index:
                continue

            test_loc = all_dates.get_loc(t_test)
            if isinstance(test_loc, slice):
                continue

            if test_loc < self.min_required_history:
                continue

            available_outer_train_dates = all_dates[:test_loc]
            if len(available_outer_train_dates) < self.outer_train_size:
                continue

            outer_train_dates = available_outer_train_dates[-self.outer_train_size:]

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
                logger.info(f"Evaluating threshold {c} for test date {t_test}")
                for theta in self.hyperparameter_grid:
                    logger.info(f"Evaluating hyperparameters {theta} for threshold {c} and test date {t_test}")
                    cv_scores = []

                    for inner_train_dates, inner_val_dates in inner_folds:
                        inner_train_returns = returns.loc[inner_train_dates]
                        inner_train_target = target.loc[inner_train_dates]

                        X_train = self.feature_pipeline.make_features(
                            returns=inner_train_returns,
                            threshold=c,
                        )
                        X_train, y_train = align_x_y(X_train, inner_train_target)

                        X_val = self.feature_pipeline.make_features_for_dates(
                            returns=returns.loc[returns.index <= inner_val_dates[-1]],
                            target_dates=inner_val_dates,
                            threshold=c,
                        )
                        y_val = target.loc[inner_val_dates]
                        X_val, y_val = align_x_y(X_val, y_val)

                        if len(X_train) == 0 or len(X_val) == 0:
                            continue

                        candidate_model = clone(self.model).set_params(**theta)
                        candidate_model.fit(X_train, y_train)

                        yhat_val = predict_positive_class_proba(candidate_model, X_val)
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
                    "scheme": "nested",
                    "outer_train_start": outer_train_dates[0],
                    "outer_train_end": outer_train_dates[-1],
                    "best_threshold": float(best_threshold),
                    "best_theta": best_theta,
                    "inner_cv_score": float(best_score),
                    "n_outer_train_obs": int(len(X_outer_train)),
                }
            )

            self.feature_pipeline.evict_before(outer_train_dates[0])

        predictions = pd.DataFrame(prediction_records).set_index("date").sort_index()
        selection_history = pd.DataFrame(selection_records).set_index("date").sort_index()

        return WalkForwardResult(
            predictions=predictions,
            selection_history=selection_history,
        )


def get_cv_runner(
    cv_scheme: CVScheme,
    feature_pipeline: RollingNetworkFeaturePipeline,
    model: Any,
    threshold_grid: Sequence[float],
    hyperparameter_grid: Sequence[Dict[str, Any]],
    scoring_func: Callable,
    outer_train_size: int,
    val_size: Optional[int] = None,
    inner_train_size: Optional[int] = None,
    inner_val_size: Optional[int] = None,
    inner_step_size: Optional[int] = 1,
    min_required_history: Optional[int] = None,
):
    """
    Factory function returning the chosen CV runner.

    Parameters
    ----------
    cv_scheme : {"simple", "nested"}
        CV scheme to use.
    feature_pipeline : RollingNetworkFeaturePipeline
        Pipeline to generate features from returns.
    model : Any
        Scikit-learn compatible model with fit/predict methods.
    threshold_grid : Sequence[float]
        Grid of thresholds for graph construction to search over.
    hyperparameter_grid : Sequence[Dict[str, Any]]
        Grid of model hyperparameters to search over.
    scoring_func : Callable
        Function to evaluate predictions against true labels for model selection.
    outer_train_size : int
        Number of observations in the outer training set.
    val_size : Optional[int]
        Number of observations in the validation set (required for "simple" scheme).
    inner_train_size : Optional[int]
        Number of observations in the inner training set (required for "nested" scheme).
    inner_val_size : Optional[int]
        Number of observations in the inner validation set (required for "nested" scheme).
    inner_step_size : Optional[int]
        Step size for rolling inner CV folds (only for "nested" scheme, default=1).
    min_required_history : Optional[int]
        Minimum number of historical observations required to run CV for a test date.
         If None, it will be set to the maximum lookback required by the feature pipeline and the training/validation sizes.
         This is used to skip test dates that don't have enough history for the CV procedure.
    """

    if cv_scheme == "simple":
        if val_size is None:
            raise ValueError("For cv_scheme='simple', val_size must be provided.")

        return SimpleRollingWalkForwardCV(
            feature_pipeline=feature_pipeline,
            model=model,
            threshold_grid=threshold_grid,
            hyperparameter_grid=hyperparameter_grid,
            scoring_func=scoring_func,
            outer_train_size=outer_train_size,
            val_size=val_size,
            min_required_history=min_required_history,
        )

    if cv_scheme == "nested":
        if inner_train_size is None or inner_val_size is None:
            raise ValueError(
                "For cv_scheme='nested', inner_train_size and inner_val_size must be provided."
            )

        return NestedWalkForwardCV(
            feature_pipeline=feature_pipeline,
            model=model,
            threshold_grid=threshold_grid,
            hyperparameter_grid=hyperparameter_grid,
            scoring_func=scoring_func,
            outer_train_size=outer_train_size,
            inner_train_size=inner_train_size,
            inner_val_size=inner_val_size,
            inner_step_size=inner_step_size,
            min_required_history=min_required_history,
        )

    raise ValueError("cv_scheme must be either 'simple' or 'nested'.")