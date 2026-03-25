"""
DataLoader class for loading and preprocessing financial data for NLP-based quantitative strategies.
"""
import pandas as pd
from stock_mkt_network_analysis.utils.config import Config
import logging
from pathlib import Path
from better_aws import AWS

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, config: Config):
        self.config = config
        self._data_attrs = []
        self.aws: AWS | None = None
        self.dates: list | None = None
        self.asset_ids: list | None = None

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

        # self._set_dates()
        # self._set_asset_ids()

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
