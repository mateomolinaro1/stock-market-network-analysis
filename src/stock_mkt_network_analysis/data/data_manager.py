"""
DataLoader class for loading and preprocessing financial data for NLP-based quantitative strategies.
"""
import pandas as pd
from stock_mkt_network_analysis.utils.config import Config
import logging
from pathlib import Path
from better_aws import AWS
from stock_mkt_network_analysis.utils.market_metric_utils import Metrics

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, config: Config):
        self.config = config
        self._data_attrs = []
        self.aws: AWS | None = None
        self.dates: list[pd.Timestamp] | None = None
        self.universe: dict[pd.Timestamp, list[int]] | None = None
        self.asset_returns: pd.DataFrame | None = None
        self.idio_asset_returns: pd.DataFrame | None = None
        self.mkt_returns: pd.DataFrame | None = None
        self.rf_returns: pd.DataFrame | None = None
        self.mkt_cumulative_returns: pd.DataFrame | None = None
        self.rolling_raw_target_variable: pd.DataFrame | None = None
        self.target_variable: pd.DataFrame | None = None
        self.target_variable_to_predict: pd.DataFrame | None = None

        # Data for analysis
        self.aligned_df: pd.DataFrame | None = None

        for filename in self.config.filenames_to_load:
            if filename == "data/wrds_funda_gross_query.parquet":
                continue
            if not isinstance(filename, str):
                logger.error(f"Invalid filename in config: {filename}. Expected a string.")
                raise ValueError(f"Invalid filename in config: {filename}. Expected a string.")

            name = Path(filename).stem

            if hasattr(self, name):
                raise ValueError(f"Duplicate attribute name generated: {name}")

            setattr(self, name, None)
            self._data_attrs.append(name)

    def load_data(self):
        if self.aws is None:
            self._init_s3()

        for filename in self.config.filenames_to_load:
            name = Path(filename).stem
            obj = self.aws.s3.load(key=filename)
            setattr(self, name, obj)

        self._build_universe()
        self._build_asset_returns()
        self._build_mkt_returns()
        self._build_rf_returns()
        if self.config.returns_type == "idio":
            self._build_idio_returns()
        self._build_target_variable()
        self._build_aligned_df()

    # ---------- Internal helpers ---------- #
    def _init_s3(self) -> None:
        """
        Initialize the AWS client and configure S3 defaults.
        """
        self.aws = AWS(region=self.config.region, verbose=True)
        # Optional sanity check
        self.aws.identity(print_info=True)
        # 2) Configure S3 defaults
        self.aws.s3.config(
            bucket=self.config.bucket_name,
            output_type="pandas",  # tabular loads -> pandas (or "polars")
            file_type="parquet",  # default tabular format for dataframe uploads without extension
            overwrite=True,
        )

    def _get_asset_attribute(self) -> pd.DataFrame:
        filename = self.config.dates_filename
        attr_name = Path(filename).stem

        if not hasattr(self, attr_name):
            raise AttributeError(f"Attribute '{attr_name}' not found. Did you load the data?")

        asset_returns_obj = getattr(self, attr_name)
        return asset_returns_obj

    def _get_mkt_attribute(self) -> pd.DataFrame:
        filename = self.config.mkt_returns_filename
        attr_name = Path(filename).stem

        if not hasattr(self, attr_name):
            raise AttributeError(f"Attribute '{attr_name}' not found. Did you load the data?")

        mkt_returns_obj = getattr(self, attr_name)
        return mkt_returns_obj

    def _get_rf_attribute(self) -> pd.DataFrame:
        filename = self.config.rf_returns_filename
        attr_name = Path(filename).stem

        if not hasattr(self, attr_name):
            raise AttributeError(f"Attribute '{attr_name}' not found. Did you load the data?")

        rf_returns_obj = getattr(self, attr_name)
        return rf_returns_obj

    def _set_dates(self) -> None:
        """
        Set self.dates from the index of the configured asset returns df.
        """

        asset_returns_obj = self._get_asset_attribute()

        if not hasattr(asset_returns_obj, "index"):
            raise TypeError("Attribute containing the asset returns has no index attribute.")

        self.dates = asset_returns_obj.index.to_list()

    def _set_asset_ids(self) -> None:
        """
        Set self.asset_ids from the columns of the configured asset returns df.
        """
        asset_returns_obj = self._get_asset_attribute()

        if not hasattr(asset_returns_obj, "columns"):
            raise TypeError("Attribute containing the asset returns has no columns attribute.")

        self.asset_ids = asset_returns_obj.columns.to_list()

    def _build_universe(self)->None:
        """
        Build the universe attribute from the loaded data.
        :return:
        """
        asset_returns = self._get_asset_attribute()
        self.dates = sorted(asset_returns['date'].unique().tolist())
        self.universe = (
            asset_returns
            .sort_values(['date', 'permno'])
            .groupby('date')['permno']
            .apply(list)
            .to_dict()
        )

    def _build_asset_returns(self) -> None:
        """
        Build the asset returns attribute (dataframe with dates as index, asset in columns and returns as values) from
        the loaded data.
        :return:
        """
        asset_returns = self._get_asset_attribute()
        asset_returns_pivot = asset_returns.pivot(index='date', columns='permno', values='ret')
        self.asset_returns = asset_returns_pivot

    def _build_mkt_returns(self) -> None:
        """
        Build the market returns attribute (dataframe with dates as index and a single column 'mkt_return' as values)
        from the loaded data.
        :return:
        """
        mkt_returns = self._get_mkt_attribute()
        mkt_returns.index = pd.to_datetime(mkt_returns.index)
        col = self.config.mkt_benchmark_column
        if col is not None:
            if col not in mkt_returns.columns:
                raise ValueError(f"MKT_BENCHMARK_COLUMN '{col}' not found in market returns. Available: {list(mkt_returns.columns)}")
            mkt_returns = mkt_returns[[col]].copy()
        else:
            mkt_returns = mkt_returns[[mkt_returns.columns[0]]].copy()
        self.mkt_returns = mkt_returns.pct_change(fill_method=None).dropna()
        cum_mkt_returns = Metrics.compute_cumulative_return(df=self.mkt_returns)
        self.mkt_cumulative_returns = cum_mkt_returns

    def _build_rf_returns(self) -> None:
        """
        Build the rf returns attribute from the loaded data.
        :return:
        """
        rf_returns = self._get_rf_attribute()
        rf_returns.index = pd.to_datetime(rf_returns.index)
        if rf_returns.index.duplicated().any():
            rf_returns = rf_returns[~rf_returns.index.duplicated(keep="last")]
        self.rf_returns = rf_returns

    def _build_idio_returns(self) -> None:
        """
        Compute CAPM idiosyncratic returns for each asset using rolling betas.
        beta_i(t) = rolling_cov(R_i - Rf, R_m - Rf) / rolling_var(R_m - Rf)
        idio_i(t) = (R_i(t) - Rf(t)) - beta_i(t) * (R_m(t) - Rf(t))
        The rolling window matches lookback_corr to ensure no look-ahead bias.
        Aligned to asset_returns.index (= self.dates) so all series share the same master index.
        """
        rf_raw = self.rf_returns
        rf = rf_raw.iloc[:, 0]
        rf.index = pd.to_datetime(rf.index)
        rf = rf[~rf.index.duplicated(keep="last")]
        rf = rf.reindex(self.asset_returns.index).ffill(limit=self.config.limit_ffill_rf)

        base_index = self.asset_returns.index

        # Rolling computation requires aligned, gap-free data — use intersection.
        # The result is then re indexed to base_index so idio_asset_returns shares
        # the same master date index as asset_returns for aligned_df merging.
        common_index = base_index.intersection(rf.index).intersection(self.mkt_returns.index)

        rf_c = rf.reindex(common_index)
        mkt_c = self.mkt_returns.iloc[:, 0].reindex(common_index)
        asset_c = self.asset_returns.reindex(common_index)

        excess_asset = asset_c.subtract(rf_c, axis=0)
        excess_mkt = mkt_c - rf_c

        window = self.config.lookback_corr
        rolling_cov = excess_asset.rolling(window).cov(excess_mkt)
        rolling_var = excess_mkt.rolling(window).var()
        beta = rolling_cov.div(rolling_var, axis=0)
        beta.ffill(inplace=True, limit=self.config.limit_ffill_betas)

        idio = excess_asset - beta.mul(excess_mkt, axis=0)
        self.idio_asset_returns = idio.reindex(base_index)

    @property
    def network_returns(self) -> pd.DataFrame:
        """Returns the returns used for network construction, sourced from aligned_df."""
        if self.aligned_df is None:
            raise ValueError("Data has not been loaded. Call load_data() first.")
        if self.config.returns_type == "idio":
            idio_cols = [c for c in self.aligned_df.columns if str(c).startswith("idio_")]
            return self.aligned_df[idio_cols].rename(columns=lambda c: int(str(c)[len("idio_"):])).dropna(how="all")
        raw_cols = [c for c in self.aligned_df.columns if c in self.asset_returns.columns]
        return self.aligned_df[raw_cols]

    def _build_target_variable(self) -> None:
        """
        Build the target variable attribute according to the values target_variable in config. This variable is
        comprised among: 'realized_volatility', 'dummy_realized_volatility', 'maximum_drawdown',
        'dummy_maximum_drawdown'.
        :return:
        """
        if self.config.target_variable == 'realized_volatility':
            self.rolling_raw_target_variable = Metrics.compute_realized_volatility(
                df=self.mkt_returns,
                rolling_window=self.config.target_variable_rolling_window,
                data_freq=self.config.data_freq
            )
            tmp = self.rolling_raw_target_variable.copy()
            self.target_variable = tmp

        elif self.config.target_variable == 'dummy_realized_volatility':
            self.rolling_raw_target_variable = Metrics.compute_realized_volatility(
                df=self.mkt_returns,
                rolling_window=self.config.target_variable_rolling_window,
                data_freq=self.config.data_freq
            )
            self.target_variable = Metrics.compute_dummy_from_feature(
                df=self.mkt_returns,
                rolling_window=self.config.target_variable_rolling_window,
                feature_func=Metrics.compute_realized_volatility,
                quantile=self.config.quantile_for_dummy
            )

        elif self.config.target_variable == 'maximum_drawdown':
            self.rolling_raw_target_variable = Metrics.compute_maximum_drawdown(
                df=self.mkt_returns,
                rolling_window=self.config.target_variable_rolling_window
            )
            tmp = self.rolling_raw_target_variable.copy()
            self.target_variable = tmp

        elif self.config.target_variable == 'dummy_maximum_drawdown':
            self.rolling_raw_target_variable = Metrics.compute_maximum_drawdown(
                df=self.mkt_returns,
                rolling_window=self.config.target_variable_rolling_window
            )
            self.target_variable = Metrics.compute_dummy_from_feature(
                df=self.mkt_returns,
                rolling_window=self.config.target_variable_rolling_window,
                feature_func=Metrics.compute_maximum_drawdown,
                quantile=self.config.quantile_for_dummy
            )
        elif self.config.target_variable == 'dummy_forward_max_drawdown':
            self.rolling_raw_target_variable = Metrics.compute_forward_max_drawdown(
                df=self.mkt_returns,
                rolling_window=self.config.target_variable_rolling_window,
            )
            self.target_variable = Metrics.compute_dummy_from_feature(
                df=self.mkt_returns,
                rolling_window=self.config.target_variable_rolling_window,
                feature_func=Metrics.compute_forward_max_drawdown,
                quantile=self.config.quantile_for_dummy
            )

        else:
            logger.error(f"Invalid target variable specified in config: {self.config.target_variable}")
            raise ValueError(f"Invalid target variable specified in config: {self.config.target_variable}")

    def _build_aligned_df(self) -> None:
        """
        Build self.aligned_df, a dataframe indexed by self.dates and containing
        the target variable, rolling raw target variable, market returns, and
        asset returns aligned on the same dates.

        For each date in self.dates, the function keeps the most recent available
        observation in each dataframe using a backward merge_asof.
        """
        base_index_name = "date"

        # Base dataframe from self.dates
        df = pd.DataFrame(index=pd.Index(self.dates, name=base_index_name)).reset_index()
        df = df.sort_values(base_index_name)

        def _merge_one(left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
            right = right_df.copy()

            # Make sure the right index has a usable name
            right.index = pd.Index(right.index, name=base_index_name)

            # Reset index and sort for merge_asof
            right = right.reset_index().sort_values(base_index_name)

            # Convert both date columns to datetime to avoid dtype mismatch
            left_df[base_index_name] = pd.to_datetime(left_df[base_index_name]).astype('datetime64[ns]')
            right[base_index_name] = pd.to_datetime(right[base_index_name]).astype('datetime64[ns]')

            merged = pd.merge_asof(
                left=left_df,
                right=right,
                on=base_index_name,
                direction="backward"
            )
            return merged

        df = _merge_one(df, self.target_variable)
        df = _merge_one(df, self.rolling_raw_target_variable)
        df = _merge_one(df, self.mkt_returns)
        df = _merge_one(df, self.mkt_cumulative_returns)
        df = _merge_one(df, self.asset_returns)
        if self.idio_asset_returns is not None:
            idio_renamed = self.idio_asset_returns.rename(columns=lambda c: f"idio_{c}")
            df = _merge_one(df, idio_renamed)

        self.aligned_df = df.set_index(base_index_name)
        # Get the first index when the target is available
        first_idx = self.aligned_df[self.config.target_variable].first_valid_index()
        cropped = self.aligned_df.loc[first_idx:,:]
        self.aligned_df = cropped
        self._shift_target_variable()

        return

    def _shift_target_variable(self) -> None:
        """
        Shift the target variable by the forecast horizon specified in config to create the target variable to predict.
        """
        if self.target_variable is None:
            raise ValueError("Target variable must be built before shifting.")

        self.target_variable_to_predict = self.aligned_df[[self.config.target_variable]].copy()
        self.target_variable_to_predict = self.target_variable_to_predict.shift(-self.config.forecasting_horizon)


