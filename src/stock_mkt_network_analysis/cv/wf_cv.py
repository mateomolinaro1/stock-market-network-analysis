"""
Walk-forward cross-validation schemes for time series classification
with a rolling network feature pipeline.

Supported schemes:
- simple: rolling train / validation / test
- nested: outer test loop + inner rolling CV for model selection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
import logging
from tqdm.auto import tqdm

from stock_mkt_network_analysis.cv.folds import build_rolling_time_series_folds
from stock_mkt_network_analysis.cv.feature_pipeline import RollingNetworkFeaturePipeline
from stock_mkt_network_analysis.models.utils import predict_positive_class_proba
from stock_mkt_network_analysis.utils.cv_cache import compute_cache_key, load_cv_result, save_cv_result
from stock_mkt_network_analysis.utils.ml_metrics import compute_oos_metrics
from stock_mkt_network_analysis.utils.utils import align_x_y


CVScheme = Literal["simple", "nested"]
logger = logging.getLogger(__name__)


def _model_name(model: Any) -> str:
    """Return the classifier class name, unwrapping sklearn Pipeline if needed."""
    if isinstance(model, Pipeline):
        return type(model.steps[-1][1]).__name__
    return type(model).__name__

@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    selection_history: pd.DataFrame
    oos_metrics: pd.DataFrame


@dataclass
class SimpleRollingWalkForwardCV:
    """
    For each test date, each model independently selects its best (threshold, params)
    via the validation window, refits on train+val, and predicts on the test date.
    All models are tracked — predictions has one row per (date, model).
    """
    feature_pipeline: RollingNetworkFeaturePipeline
    model_grid: Sequence[Tuple[Any, Sequence[Dict[str, Any]]]]  # [(estimator, param_grid), ...]
    threshold_grid: Sequence[float]
    scoring_func: Callable
    outer_train_size: int
    val_size: int
    target_horizon: int = 0
    min_required_history: Optional[int] = None
    feature_mode: str = "all"
    expanding_or_rolling: str = "rolling"
    evict_cache: bool = True

    def run(
        self,
        returns: pd.DataFrame,
        target: pd.Series,
        outer_test_dates: Sequence[pd.Timestamp],
        aws_s3=None,
        cv_config=None,
    ) -> WalkForwardResult:

        cache_key = None
        if aws_s3 is not None and cv_config is not None:
            cache_key = compute_cache_key(cv_config)
            logger.info(f"CV cache key: {cache_key}")
            if cv_config.load_or_compute_cv == "load":
                cached = load_cv_result(cache_key, aws_s3, self.feature_mode)
                if cached is not None:
                    logger.info("Loaded CV result from S3 cache.")
                    return WalkForwardResult(**cached)
                logger.warning("Cache miss — computing CV from scratch.")

        returns = returns.sort_index()
        target = target.sort_index()
        outer_test_dates = pd.Index(outer_test_dates).sort_values()

        lookback = self.feature_pipeline.correlation_estimator.lookback

        if self.min_required_history is None:
            self.min_required_history = lookback + self.outer_train_size + self.val_size + self.target_horizon

        prediction_records: List[Dict[str, Any]] = []
        selection_records: List[Dict[str, Any]] = []

        all_dates = returns.index

        # Incremental feature cache: avoids rebuilding large DataFrames from scratch each step.
        # Keyed by threshold; mode is constant per runner so no need to key by it.
        # At each step we only call make_features for the *new* dates entering the window,
        # then append (and for rolling, prune) the stored DataFrame.
        _x_train: Dict[float, pd.DataFrame] = {}
        _x_refit: Dict[float, pd.DataFrame] = {}
        _prev_train_end: int = -1
        _prev_val_end: int = -1

        for t_test in tqdm(outer_test_dates, desc=f"Walk-forward CV [{self.feature_mode}]", unit="date"):
            logger.debug(f"Processing test date {t_test}")
            if t_test not in returns.index or t_test not in target.index:
                continue

            test_loc = all_dates.get_loc(t_test)
            if isinstance(test_loc, slice):
                continue

            if test_loc < self.min_required_history:
                continue

            # Buffer val_end back by target_horizon so the last val label
            # only uses returns up to t_test - 1 (no leakage into the test period)
            val_end = test_loc - self.target_horizon
            val_start = val_end - self.val_size
            train_end = val_start
            if self.expanding_or_rolling == "expanding":
                train_start = 0
            else:
                train_start = train_end - self.outer_train_size

            if train_start < 0 or train_end <= train_start:
                continue

            train_dates = all_dates[train_start:train_end]
            val_dates = all_dates[val_start:val_end]
            refit_dates = all_dates[train_start:val_end]
            test_dates = pd.Index([t_test])

            # ── Update incremental feature caches (once per test date, per threshold) ──
            for c in self.threshold_grid:
                # Train cache
                if c in _x_train and _prev_train_end >= 0:
                    new_train = all_dates[_prev_train_end:train_end]
                    if len(new_train) > 0:
                        new_X = self.feature_pipeline.make_features(
                            returns=returns.loc[new_train], threshold=c, mode=self.feature_mode,
                        )
                        if not new_X.empty:
                            _x_train[c] = pd.concat([_x_train[c], new_X])
                    if self.expanding_or_rolling == "rolling":
                        _x_train[c] = _x_train[c].loc[_x_train[c].index >= train_dates[0]]
                else:
                    _x_train[c] = self.feature_pipeline.make_features(
                        returns=returns.loc[train_dates], threshold=c, mode=self.feature_mode,
                    )

                # Refit cache (train + val window)
                if c in _x_refit and _prev_val_end >= 0:
                    new_refit = all_dates[_prev_val_end:val_end]
                    if len(new_refit) > 0:
                        new_Xr = self.feature_pipeline.make_features(
                            returns=returns.loc[new_refit], threshold=c, mode=self.feature_mode,
                        )
                        if not new_Xr.empty:
                            _x_refit[c] = pd.concat([_x_refit[c], new_Xr])
                    if self.expanding_or_rolling == "rolling":
                        _x_refit[c] = _x_refit[c].loc[_x_refit[c].index >= refit_dates[0]]
                else:
                    _x_refit[c] = self.feature_pipeline.make_features(
                        returns=returns.loc[refit_dates], threshold=c, mode=self.feature_mode,
                    )

            _prev_train_end = train_end
            _prev_val_end = val_end

            # ── Val and test features: small windows, always compute fresh ──────────
            val_cache: Dict[float, tuple] = {}
            test_cache: Dict[float, tuple] = {}
            for c in self.threshold_grid:
                X_val = self.feature_pipeline.make_features_for_dates(
                    returns=returns.loc[returns.index <= val_dates[-1]],
                    target_dates=val_dates, threshold=c, mode=self.feature_mode,
                )
                X_val, y_val = align_x_y(X_val, target.loc[val_dates])
                val_cache[c] = (X_val, y_val)

                X_test = self.feature_pipeline.make_features_for_dates(
                    returns=returns.loc[returns.index <= t_test],
                    target_dates=test_dates, threshold=c, mode=self.feature_mode,
                )
                X_test, y_test = align_x_y(X_test, target.loc[test_dates])
                test_cache[c] = (X_test, y_test)

            # ── Model selection: feature matrices looked up from cache, not recomputed ──
            for base_model, param_grid in self.model_grid:
                model_name = _model_name(base_model)

                best_score = -np.inf
                best_threshold = None
                best_theta = None

                for c in self.threshold_grid:
                    X_train, y_train = align_x_y(_x_train[c], target.loc[train_dates])
                    X_val, y_val = val_cache[c]

                    if len(X_train) == 0 or len(X_val) == 0:
                        continue

                    for theta in param_grid:
                        if len(np.unique(y_train)) < 2:
                            fitted = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
                        else:
                            fitted = clone(base_model).set_params(**theta)
                            fitted.fit(X_train, y_train)

                        yhat_val = predict_positive_class_proba(fitted, X_val)
                        score = self.scoring_func(y_val.values, yhat_val)

                        if not np.isnan(score) and score > best_score:
                            best_score = score
                            best_threshold = c
                            best_theta = theta

                if best_threshold is None or best_theta is None:
                    logger.warning(f"No valid combo found for {model_name} at {t_test} — using DummyClassifier fallback.")
                    y_refit_full = target.loc[refit_dates].dropna()
                    y_test_full = target.loc[test_dates].dropna()
                    if len(y_refit_full) == 0 or len(y_test_full) == 0:
                        continue
                    dummy_X_refit = np.zeros((len(y_refit_full), 1))
                    dummy_X_test = np.zeros((len(y_test_full), 1))
                    final_model = DummyClassifier(strategy="most_frequent").fit(dummy_X_refit, y_refit_full)
                    yhat_test = predict_positive_class_proba(final_model, dummy_X_test)
                    prediction_records.append({
                        "date": t_test,
                        "model": model_name,
                        "feature_mode": self.feature_mode,
                        "y_true": int(y_test_full.iloc[0]),
                        "y_pred": float(yhat_test[0]),
                        "y_pred_binary": int(yhat_test[0] >= 0.5),
                        "best_threshold": float("nan"),
                        "validation_score": float("nan"),
                    })
                    selection_records.append({
                        "date": t_test,
                        "model": model_name,
                        "feature_mode": self.feature_mode,
                        "scheme": "simple",
                        "train_start": train_dates[0],
                        "train_end": train_dates[-1],
                        "val_start": val_dates[0],
                        "val_end": val_dates[-1],
                        "best_threshold": float("nan"),
                        "best_theta": None,
                        "validation_score": float("nan"),
                        "n_refit_obs": int(len(y_refit_full)),
                    })
                    continue

                X_refit, y_refit = align_x_y(_x_refit[best_threshold], target.loc[refit_dates])
                X_test, y_test = test_cache[best_threshold]

                if len(X_refit) == 0 or len(X_test) == 0:
                    continue

                if len(np.unique(y_refit)) < 2:
                    final_model = DummyClassifier(strategy="most_frequent").fit(X_refit, y_refit)
                else:
                    final_model = clone(base_model).set_params(**best_theta)
                    final_model.fit(X_refit, y_refit)

                yhat_test = predict_positive_class_proba(final_model, X_test)

                prediction_records.append(
                    {
                        "date": t_test,
                        "model": model_name,
                        "feature_mode": self.feature_mode,
                        "y_true": int(y_test.iloc[0]),
                        "y_pred": float(yhat_test[0]),
                        "y_pred_binary": int(yhat_test[0] >= 0.5),
                        "best_threshold": float(best_threshold),
                        "validation_score": float(best_score),
                    }
                )

                selection_records.append(
                    {
                        "date": t_test,
                        "model": model_name,
                        "feature_mode": self.feature_mode,
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

            if self.evict_cache:
                self.feature_pipeline.evict_before(train_dates[0])

        model_names = [_model_name(base_model) for base_model, _ in self.model_grid]
        full_index = pd.MultiIndex.from_product(
            [outer_test_dates, model_names, [self.feature_mode]],
            names=["date", "model", "feature_mode"],
        )

        predictions = (
            pd.DataFrame(prediction_records)
            .set_index(["date", "model", "feature_mode"])
            .reindex(full_index)
            .sort_index()
        )
        selection_history = (
            pd.DataFrame(selection_records)
            .set_index(["date", "model", "feature_mode"])
            .reindex(full_index)
            .sort_index()
        )

        result = WalkForwardResult(
            predictions=predictions,
            selection_history=selection_history,
            oos_metrics=compute_oos_metrics(predictions),
        )

        if aws_s3 is not None and cv_config is not None and cv_config.save_cv:
            save_cv_result(result, cache_key, aws_s3, self.feature_mode)

        return result


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
    target_horizon: int = 0
    inner_step_size: Optional[int] = 1
    min_required_history: Optional[int] = None

    def run(
        self,
        returns: pd.DataFrame,
        target: pd.Series,
        outer_test_dates: Sequence[pd.Timestamp],
        aws_s3=None,
        cv_config=None,
    ) -> WalkForwardResult:

        cache_key = None
        if aws_s3 is not None and cv_config is not None:
            cache_key = compute_cache_key(cv_config)
            logger.info(f"CV cache key: {cache_key}")
            if cv_config.load_or_compute_cv == "load":
                cached = load_cv_result(cache_key, aws_s3)
                if cached is not None:
                    logger.info("Loaded CV result from S3 cache.")
                    return WalkForwardResult(**cached)
                logger.warning("Cache miss — computing CV from scratch.")

        returns = returns.sort_index()
        target = target.sort_index()
        outer_test_dates = pd.Index(outer_test_dates).sort_values()

        lookback = self.feature_pipeline.correlation_estimator.lookback

        if self.min_required_history is None:
            self.min_required_history = max(
                lookback + self.outer_train_size,
                lookback + self.inner_train_size + self.inner_val_size + self.target_horizon,
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

            # Trim last target_horizon dates so inner val labels don't leak into t_test
            inner_fold_dates = outer_train_dates[:-self.target_horizon] if self.target_horizon > 0 else outer_train_dates
            inner_folds = build_rolling_time_series_folds(
                dates=inner_fold_dates,
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

                        if len(np.unique(y_train)) < 2:
                            candidate_model = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
                        else:
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
                logger.warning(f"No valid combo found for test date {t_test} — using DummyClassifier fallback.")
                y_outer_train = target.loc[outer_train_dates].dropna()
                y_outer_test = target.loc[[t_test]].dropna()
                if len(y_outer_train) == 0 or len(y_outer_test) == 0:
                    continue
                dummy_X_train = np.zeros((len(y_outer_train), 1))
                dummy_X_test = np.zeros((len(y_outer_test), 1))
                final_model = DummyClassifier(strategy="most_frequent").fit(dummy_X_train, y_outer_train)
                yhat_test = predict_positive_class_proba(final_model, dummy_X_test)
                prediction_records.append({
                    "date": t_test,
                    "y_true": int(y_outer_test.iloc[0]),
                    "y_pred": float(yhat_test[0]),
                    "y_pred_binary": int(yhat_test[0] >= 0.5),
                    "best_threshold": float("nan"),
                    "inner_cv_score": float("nan"),
                })
                selection_records.append({
                    "date": t_test,
                    "scheme": "nested",
                    "outer_train_start": outer_train_dates[0],
                    "outer_train_end": outer_train_dates[-1],
                    "best_threshold": float("nan"),
                    "best_theta": None,
                    "inner_cv_score": float("nan"),
                    "n_outer_train_obs": int(len(y_outer_train)),
                })
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

            if len(np.unique(y_outer_train)) < 2:
                final_model = DummyClassifier(strategy="most_frequent").fit(X_outer_train, y_outer_train)
            else:
                final_model = clone(self.model).set_params(**best_theta)
                final_model.fit(X_outer_train, y_outer_train)

            yhat_test = predict_positive_class_proba(final_model, X_outer_test)

            prediction_records.append(
                {
                    "date": t_test,
                    "y_true": int(y_outer_test.iloc[0]),
                    "y_pred": float(yhat_test[0]),
                    "y_pred_binary": int(yhat_test[0] >= 0.5),
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

        result = WalkForwardResult(
            predictions=predictions,
            selection_history=selection_history,
            oos_metrics=compute_oos_metrics(predictions),
        )

        if aws_s3 is not None and cv_config is not None and cv_config.save_cv:
            save_cv_result(result, cache_key, aws_s3)

        return result


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
    target_horizon: int = 0,
    min_required_history: Optional[int] = None,
):
    if cv_scheme == "simple":
        if val_size is None:
            raise ValueError("For cv_scheme='simple', val_size must be provided.")

        return SimpleRollingWalkForwardCV(
            feature_pipeline=feature_pipeline,
            model_grid=[(model, hyperparameter_grid)],
            threshold_grid=threshold_grid,
            scoring_func=scoring_func,
            outer_train_size=outer_train_size,
            val_size=val_size,
            target_horizon=target_horizon,
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
            target_horizon=target_horizon,
            inner_step_size=inner_step_size,
            min_required_history=min_required_history,
        )

    raise ValueError("cv_scheme must be either 'simple' or 'nested'.")