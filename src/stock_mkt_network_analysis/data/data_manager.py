"""
DataLoader class for loading and preprocessing financial data for NLP-based quantitative strategies.
"""
import pandas as pd
from stock_mkt_network_analysis.utils.config import Config
import logging
from pathlib import Path
from better_aws import AWS
from stock_mkt_network_analysis.utils.metric_utils import Metrics

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, config: Config):
        self.config = config
        self._data_attrs = []
        self.aws: AWS | None = None
        self.dates: list[pd.Timestamp] | None = None
        self.universe: dict[pd.Timestamp, list[int]] | None = None
        self.asset_returns: pd.DataFrame | None = None
        self.mkt_returns: pd.DataFrame | None = None
        self.mkt_cumulative_returns: pd.DataFrame | None = None
        self.rolling_raw_target_variable: pd.DataFrame | None = None
        self.target_variable: pd.DataFrame | None = None

        # Data for analysis
        self.aligned_df: pd.DataFrame | None = None

        for filename in self.config.filenames_to_load:
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
        mkt_returns = mkt_returns[[mkt_returns.columns[0]]].copy()
        self.mkt_returns = mkt_returns.pct_change(fill_method=None).dropna()
        cum_mkt_returns = Metrics.compute_cumulative_return(df=self.mkt_returns)
        self.mkt_cumulative_returns = cum_mkt_returns

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
            self.target_variable = Metrics.compute_realized_volatility(
                df=self.mkt_returns,
                rolling_window=self.config.target_variable_rolling_window,
                data_freq=self.config.data_freq
            )
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
            self.target_variable = Metrics.compute_maximum_drawdown(
                df=self.mkt_returns,
                rolling_window=self.config.target_variable_rolling_window
            )
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

        self.aligned_df = df.set_index(base_index_name)

        return


