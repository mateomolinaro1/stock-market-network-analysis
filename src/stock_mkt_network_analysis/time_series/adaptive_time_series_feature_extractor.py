from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_ANNUALIZATION = {
    "d": 252,
    "w": 52,
    "m": 12,
    "y": 1,
}


@dataclass
class AdaptiveTimeSeriesFeatureExtractor:
    """
    Adaptive time-series feature extractor for drawdown prediction.

    The extractor builds market-level features from asset returns, optional
    benchmark returns, optional risk-free returns, and optional volumes.

    Design principles:
    - no look-ahead: features for date t use observations strictly before t;
    - adaptive windows: short / medium / long windows are derived from lookback;
    - robust aggregation: asset-level information is summarized cross-sectionally;
    - model-ready output: one row per eligible date, one feature per column.
    """

    lookback: int
    data_freq: str = "d"
    forecasting_horizon: Optional[int] = None
    min_obs_ratio: float = 0.8
    include_asset_cross_section: bool = True
    include_distribution_features: bool = True
    include_volume_features: bool = True
    eps: float = 1e-12
    windows: Optional[Sequence[int]] = None

    def __post_init__(self) -> None:
        if self.lookback < 5:
            raise ValueError("lookback must be >= 5 to compute stable time-series features.")
        if not 0 < self.min_obs_ratio <= 1:
            raise ValueError("min_obs_ratio must be in (0, 1].")
        if self.data_freq not in _ANNUALIZATION:
            raise ValueError(f"Unsupported data_freq={self.data_freq!r}. Use one of {list(_ANNUALIZATION)}.")

        if self.windows is None:
            self.windows = self._default_windows(self.lookback, self.forecasting_horizon)
        self.windows = tuple(sorted(set(int(w) for w in self.windows if int(w) >= 2 and int(w) <= self.lookback)))

        if not self.windows:
            raise ValueError("No valid windows available. Check lookback/windows.")

    @staticmethod
    def _default_windows(lookback: int, forecasting_horizon: Optional[int]) -> Sequence[int]:
        """
        Build sensible windows from the experiment parameters.

        Examples:
        - lookback=21  -> (5, 10, 21)
        - lookback=63  -> (10, 21, 63)
        - lookback=126 -> (21, 63, 126)
        """
        candidates = {
            max(2, round(lookback / 4)),
            max(2, round(lookback / 2)),
            lookback,
        }
        if forecasting_horizon is not None:
            candidates.add(max(2, min(int(forecasting_horizon), lookback)))
        return sorted(candidates)

    @classmethod
    def from_config(cls, config: object, **overrides: object) -> "AdaptiveTimeSeriesFeatureExtractor":
        """
        Instantiate from the project's Config object.

        Expected config attributes:
        - lookback_corr or LOOKBACK
        - data_freq
        - forecasting_horizon
        """
        lookback = getattr(config, "lookback_corr", None)
        if lookback is None:
            lookback = getattr(config, "LOOKBACK", None)
        if lookback is None:
            raise ValueError("Config must expose `lookback_corr` or `LOOKBACK`.")

        kwargs = {
            "lookback": int(lookback),
            "data_freq": getattr(config, "data_freq", "d") or "d",
            "forecasting_horizon": getattr(config, "forecasting_horizon", None),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def compute_for_date(
        self,
        asset_returns: pd.DataFrame,
        date: pd.Timestamp,
        market_returns: Optional[pd.DataFrame | pd.Series] = None,
        risk_free_returns: Optional[pd.DataFrame | pd.Series] = None,
        volumes: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Compute features for one target date using only rows strictly before date.
        """
        asset_returns = self._prepare_frame(asset_returns)
        date = pd.Timestamp(date)

        if date not in asset_returns.index:
            return {}

        loc = asset_returns.index.get_loc(date)
        if isinstance(loc, slice) or loc < self.lookback:
            return {}

        window_assets = asset_returns.iloc[loc - self.lookback:loc]
        if len(window_assets) < self.lookback:
            return {}

        market_series = self._slice_optional_series(market_returns, asset_returns.index, loc)
        rf_series = self._slice_optional_series(risk_free_returns, asset_returns.index, loc)
        volume_window = self._slice_optional_frame(volumes, asset_returns.index, loc)

        return self.transform(
            asset_returns=window_assets,
            market_returns=market_series,
            risk_free_returns=rf_series,
            volumes=volume_window,
        )

    def compute_rolling(
        self,
        asset_returns: pd.DataFrame,
        market_returns: Optional[pd.DataFrame | pd.Series] = None,
        risk_free_returns: Optional[pd.DataFrame | pd.Series] = None,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute adaptive time-series features for all eligible dates.
        """
        asset_returns = self._prepare_frame(asset_returns)
        rows: Dict[pd.Timestamp, Dict[str, float]] = {}

        total = max(0, len(asset_returns) - self.lookback)
        log_every = max(1, total // 10) if total else 1
        logger.info("Starting adaptive TS features: %s dates, lookback=%s, windows=%s", total, self.lookback, self.windows)

        for i in range(self.lookback, len(asset_returns)):
            date = asset_returns.index[i]
            window_assets = asset_returns.iloc[i - self.lookback:i]
            market_series = self._slice_optional_series(market_returns, asset_returns.index, i)
            rf_series = self._slice_optional_series(risk_free_returns, asset_returns.index, i)
            volume_window = self._slice_optional_frame(volumes, asset_returns.index, i)

            rows[date] = self.transform(window_assets, market_series, rf_series, volume_window)

            step = i - self.lookback + 1
            if step % log_every == 0 or step == total:
                logger.info("  %s/%s (%s%%) — last date: %s", step, total, 100 * step // max(total, 1), date.date())

        return pd.DataFrame.from_dict(rows, orient="index").sort_index().astype("float32")

    def transform(
        self,
        asset_returns: pd.DataFrame,
        market_returns: Optional[pd.Series] = None,
        risk_free_returns: Optional[pd.Series] = None,
        volumes: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Transform a historical lookback window into model-ready features.
        """
        r = self._drop_sparse_assets(asset_returns)
        if r.empty:
            return {}

        features: Dict[str, float] = {}
        annual_factor = _ANNUALIZATION[self.data_freq]
        sqrt_ann = float(np.sqrt(annual_factor))

        # Equal-weighted market proxy when no benchmark is provided.
        mkt = self._clean_series(market_returns)
        if mkt is None or mkt.empty:
            mkt = r.mean(axis=1)
        else:
            mkt = mkt.reindex(r.index).dropna()

        if risk_free_returns is not None:
            rf = self._clean_series(risk_free_returns).reindex(mkt.index).fillna(0.0)
            excess_mkt = mkt - rf
        else:
            excess_mkt = mkt

        for w in self.windows:
            suffix = f"_{w}"
            m = excess_mkt.tail(w).dropna()
            if len(m) < max(2, int(w * self.min_obs_ratio)):
                continue

            features[f"mkt_return{suffix}"] = self._compound_return(m)
            features[f"mkt_mean_return{suffix}"] = float(m.mean())
            features[f"mkt_vol{suffix}"] = float(m.std(ddof=1) * sqrt_ann)
            features[f"mkt_downside_vol{suffix}"] = self._downside_vol(m, sqrt_ann)
            features[f"mkt_drawdown{suffix}"] = self._current_drawdown(m)
            features[f"mkt_max_drawdown{suffix}"] = self._max_drawdown(m)
            features[f"mkt_drawdown_speed{suffix}"] = self._drawdown_speed(m)
            features[f"mkt_var_05{suffix}"] = float(m.quantile(0.05))
            features[f"mkt_cvar_05{suffix}"] = self._cvar(m, 0.05)
            features[f"mkt_hit_ratio{suffix}"] = float((m > 0).mean())
            features[f"mkt_abs_return_last{suffix}"] = float(abs(m.iloc[-1]))

            if self.include_distribution_features and len(m) >= 5:
                features[f"mkt_skew{suffix}"] = float(m.skew())
                features[f"mkt_kurt{suffix}"] = float(m.kurt())

            if self.include_asset_cross_section:
                rw = r.tail(w)
                features.update(self._cross_section_features(rw, suffix, sqrt_ann))

            if self.include_volume_features and volumes is not None:
                vw = volumes.reindex(r.index).tail(w)
                features.update(self._volume_features(vw, suffix))

        features.update(self._ratio_features(features))
        return features

    def _cross_section_features(self, returns: pd.DataFrame, suffix: str, sqrt_ann: float) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if returns.empty:
            return out

        last = returns.iloc[-1].dropna()
        out[f"cs_breadth_positive{suffix}"] = float((last > 0).mean()) if len(last) else np.nan
        out[f"cs_breadth_negative{suffix}"] = float((last < 0).mean()) if len(last) else np.nan
        out[f"cs_return_dispersion_last{suffix}"] = float(last.std(ddof=1)) if len(last) > 1 else np.nan
        out[f"cs_return_p10_last{suffix}"] = float(last.quantile(0.10)) if len(last) else np.nan
        out[f"cs_return_p90_last{suffix}"] = float(last.quantile(0.90)) if len(last) else np.nan

        asset_cumret = returns.apply(self._compound_return, axis=0)
        out[f"cs_avg_asset_return{suffix}"] = float(asset_cumret.mean())
        out[f"cs_std_asset_return{suffix}"] = float(asset_cumret.std(ddof=1))
        out[f"cs_worst_asset_return{suffix}"] = float(asset_cumret.min())
        out[f"cs_best_asset_return{suffix}"] = float(asset_cumret.max())

        asset_vol = returns.std(axis=0, ddof=1) * sqrt_ann
        out[f"cs_avg_asset_vol{suffix}"] = float(asset_vol.mean())
        out[f"cs_std_asset_vol{suffix}"] = float(asset_vol.std(ddof=1))
        out[f"cs_max_asset_vol{suffix}"] = float(asset_vol.max())

        asset_mdd = returns.apply(self._max_drawdown, axis=0)
        out[f"cs_avg_asset_max_drawdown{suffix}"] = float(asset_mdd.mean())
        out[f"cs_worst_asset_max_drawdown{suffix}"] = float(asset_mdd.min())
        out[f"cs_share_assets_in_drawdown{suffix}"] = float((returns.apply(self._current_drawdown, axis=0) < 0).mean())
        return out

    def _volume_features(self, volumes: pd.DataFrame, suffix: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if volumes.empty:
            return out
        total_volume = volumes.sum(axis=1, min_count=1).dropna()
        if len(total_volume) < 2:
            return out
        out[f"volume_growth{suffix}"] = float(total_volume.iloc[-1] / (total_volume.mean() + self.eps) - 1.0)
        out[f"volume_volatility{suffix}"] = float(total_volume.pct_change().replace([np.inf, -np.inf], np.nan).std(ddof=1))
        return out

    def _ratio_features(self, features: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if len(self.windows) < 2:
            return out
        short = self.windows[0]
        long = self.windows[-1]
        for base in ["mkt_vol", "mkt_downside_vol", "cs_avg_asset_vol"]:
            a = features.get(f"{base}_{short}")
            b = features.get(f"{base}_{long}")
            if a is not None and b is not None:
                out[f"{base}_ratio_{short}_{long}"] = float(a / (b + self.eps))
        for base in ["mkt_return", "mkt_max_drawdown", "cs_breadth_positive"]:
            a = features.get(f"{base}_{short}")
            b = features.get(f"{base}_{long}")
            if a is not None and b is not None:
                out[f"{base}_delta_{short}_{long}"] = float(a - b)
        return out

    def _drop_sparse_assets(self, returns: pd.DataFrame) -> pd.DataFrame:
        min_obs = max(2, int(len(returns) * self.min_obs_ratio))
        clean = returns.dropna(axis=1, thresh=min_obs)
        return clean.astype(float)

    @staticmethod
    def _compound_return(x: pd.Series) -> float:
        x = pd.Series(x).dropna()
        if x.empty:
            return np.nan
        return float((1.0 + x).prod() - 1.0)

    @staticmethod
    def _wealth(x: pd.Series) -> pd.Series:
        return (1.0 + pd.Series(x).dropna()).cumprod()

    @classmethod
    def _current_drawdown(cls, x: pd.Series) -> float:
        wealth = cls._wealth(x)
        if wealth.empty:
            return np.nan
        return float(wealth.iloc[-1] / wealth.cummax().iloc[-1] - 1.0)

    @classmethod
    def _max_drawdown(cls, x: pd.Series) -> float:
        wealth = cls._wealth(x)
        if wealth.empty:
            return np.nan
        dd = wealth / wealth.cummax() - 1.0
        return float(dd.min())

    @classmethod
    def _drawdown_speed(cls, x: pd.Series) -> float:
        wealth = cls._wealth(x)
        if len(wealth) < 2:
            return np.nan
        dd = wealth / wealth.cummax() - 1.0
        return float(dd.iloc[-1] - dd.iloc[-2])

    @staticmethod
    def _downside_vol(x: pd.Series, sqrt_ann: float) -> float:
        neg = pd.Series(x).dropna()
        neg = neg[neg < 0]
        if len(neg) < 2:
            return 0.0
        return float(neg.std(ddof=1) * sqrt_ann)

    @staticmethod
    def _cvar(x: pd.Series, q: float) -> float:
        x = pd.Series(x).dropna()
        if x.empty:
            return np.nan
        var = x.quantile(q)
        tail = x[x <= var]
        return float(tail.mean()) if len(tail) else float(var)

    @staticmethod
    def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.index = pd.to_datetime(out.index)
        return out.sort_index().astype(float)

    @staticmethod
    def _clean_series(obj: Optional[pd.DataFrame | pd.Series]) -> Optional[pd.Series]:
        if obj is None:
            return None
        if isinstance(obj, pd.DataFrame):
            if obj.empty:
                return pd.Series(dtype=float)
            s = obj.iloc[:, 0]
        else:
            s = obj
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        return s.sort_index().astype(float)

    def _slice_optional_series(
        self,
        obj: Optional[pd.DataFrame | pd.Series],
        reference_index: pd.Index,
        loc: int,
    ) -> Optional[pd.Series]:
        s = self._clean_series(obj)
        if s is None:
            return None
        dates = reference_index[loc - self.lookback:loc]
        return s.reindex(dates)

    def _slice_optional_frame(
        self,
        obj: Optional[pd.DataFrame],
        reference_index: pd.Index,
        loc: int,
    ) -> Optional[pd.DataFrame]:
        if obj is None:
            return None
        f = obj.copy()
        f.index = pd.to_datetime(f.index)
        f = f.sort_index().astype(float)
        dates = reference_index[loc - self.lookback:loc]
        return f.reindex(dates)
