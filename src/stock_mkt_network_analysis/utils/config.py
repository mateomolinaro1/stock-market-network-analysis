from dataclasses import dataclass
from pathlib import Path
import logging
import json
from typing import Any, List, Dict, Tuple

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """
    Configuration object to hold settings for the application.
    """
    def __init__(self):
        # Paths
        try:
            self.ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
        except NameError:
            self.ROOT_DIR = Path.cwd()
        logger.info("Root dir: " + str(self.ROOT_DIR))

        self.RUN_PIPELINE_CONFIG_PATH = self.ROOT_DIR / "config" / "run_pipeline_config.json"
        logger.info("run_pipeline config path: " + str(self.RUN_PIPELINE_CONFIG_PATH))

        # AWS
        self.bucket_name: str|None = None
        self.region: str|None = None
        self.output_format: str|None = None
        self.filenames_to_load: List[str]|None = None
        self.dates_filename: str|None = None
        self.mkt_returns_filename: str|None = None
        self.rf_returns_filename: str|None = None

        # Data
        self.data_freq: str|None = None
        self.target_variable: str|None = None
        self.target_variable_rolling_window: int|None = None
        self.quantile_for_dummy: float|None = None
        self.returns_type: str = "raw"
        self.mkt_benchmark_column: str | None = None
        self.limit_ffill_betas: int|None = None
        self.limit_ffill_rf: int|None = None

        # Forecasting
        self.forecasting_horizon: int|None = None
        self.lookback_corr: int|None = None
        self.inner_train_size: int|None = None
        self.inner_val_size: int|None = None
        self.inner_step_size: int|None = None
        self.threshold_grid: List[float]|None = None
        self.logit_param_grid: List[Dict]|None = None
        self.model_grid: List[Tuple[Any, List[Dict]]]|None = None
        self.scoring_metric: str = "roc_auc"
        self.load_or_compute_cv: str = "compute"
        self.save_cv: bool = False

        # Load JSON config to attributes of Config class
        self._load_run_pipeline_config()

    _MODEL_REGISTRY = {
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
    }

    @staticmethod
    def _parse_model_grid(raw: list) -> List[Tuple[Any, List[Dict]]]:
        result = []
        for entry in raw:
            cls = Config._MODEL_REGISTRY[entry["model"]]
            estimator = cls(**entry.get("model_kwargs", {}))
            result.append((estimator, entry["param_grid"]))
        return result

    def _load_run_pipeline_config(self)->None:
        """
        Load run_pipeline_config.json file
        :return:
        """
        with open(self.ROOT_DIR / "config" / "run_pipeline_config.json" , "r") as f:
            config: dict = json.load(f)

            # AWS
            if config.get("AWS").get("S3").get("BUCKET_NAME") is not None:
                self.bucket_name = config.get("AWS").get("S3").get("BUCKET_NAME")
            if config.get("AWS").get("S3").get("AWS_DEFAULT_REGION") is not None:
                self.region = config.get("AWS").get("S3").get("AWS_DEFAULT_REGION")
            if config.get("AWS").get("S3").get("OUTPUT_FORMAT") is not None:
                self.output_format = config.get("AWS").get("S3").get("OUTPUT_FORMAT")
            if config.get("AWS").get("S3").get("FILENAMES_TO_LOAD") is not None:
                self.filenames_to_load = config.get("AWS").get("S3").get("FILENAMES_TO_LOAD")
            if config.get("AWS").get("S3").get("DATES_FILENAME") is not None:
                self.dates_filename = config.get("AWS").get("S3").get("DATES_FILENAME")
            if config.get("AWS").get("S3").get("MKT_RETURNS_FILENAME") is not None:
                self.mkt_returns_filename = config.get("AWS").get("S3").get("MKT_RETURNS_FILENAME")
                if config.get("AWS").get("S3").get("RF_RETURNS_FILENAME") is not None:
                    self.rf_returns_filename = config.get("AWS").get("S3").get("RF_RETURNS_FILENAME")

            # Data
            if config.get("DATA").get("DATA_FREQ") is not None:
                self.data_freq = config.get("DATA").get("DATA_FREQ")
            if config.get("DATA").get("TARGET_VARIABLE") is not None:
                self.target_variable = config.get("DATA").get("TARGET_VARIABLE")
            if config.get("DATA").get("TARGET_VARIABLE_ROLLING_WINDOW") is not None:
                self.target_variable_rolling_window = config.get("DATA").get("TARGET_VARIABLE_ROLLING_WINDOW")
            if config.get("DATA").get("QUANTILE_FOR_DUMMY") is not None:
                self.quantile_for_dummy = config.get("DATA").get("QUANTILE_FOR_DUMMY")
            if config.get("DATA").get("RETURNS_TYPE") is not None:
                self.returns_type = config.get("DATA").get("RETURNS_TYPE")
            if config.get("DATA").get("MKT_BENCHMARK_COLUMN") is not None:
                self.mkt_benchmark_column = config.get("DATA").get("MKT_BENCHMARK_COLUMN")
            if config.get("DATA").get("LIMIT_FFILL_BETAS") is not None:
                self.limit_ffill_betas = config.get("DATA").get("LIMIT_FFILL_BETAS")
            if config.get("DATA").get("LIMIT_FFILL_RF") is not None:
                self.limit_ffill_rf = config.get("DATA").get("LIMIT_FFILL_RF")

            # Forecasting
            forecasting = config.get("FORECASTING", {})
            if forecasting.get("FORECASTING_HORIZON") is not None:
                self.forecasting_horizon = forecasting.get("FORECASTING_HORIZON")
            if forecasting.get("LOOKBACK_CORR") is not None:
                self.lookback_corr = forecasting.get("LOOKBACK_CORR")
            if forecasting.get("INNER_TRAIN_SIZE") is not None:
                self.inner_train_size = forecasting.get("INNER_TRAIN_SIZE")
            if forecasting.get("INNER_VAL_SIZE") is not None:
                self.inner_val_size = forecasting.get("INNER_VAL_SIZE")
            if forecasting.get("INNER_STEP_SIZE") is not None:
                self.inner_step_size = forecasting.get("INNER_STEP_SIZE")
            if forecasting.get("THRESHOLD_GRID") is not None:
                self.threshold_grid = forecasting.get("THRESHOLD_GRID")
            if forecasting.get("LOGIT_PARAM_GRID") is not None:
                self.logit_param_grid = forecasting.get("LOGIT_PARAM_GRID")
            if forecasting.get("MODEL_GRID") is not None:
                self.model_grid = self._parse_model_grid(forecasting.get("MODEL_GRID"))
            if forecasting.get("SCORING_METRIC") is not None:
                self.scoring_metric = forecasting.get("SCORING_METRIC")
            if forecasting.get("LOAD_OR_COMPUTE_CV") is not None:
                self.load_or_compute_cv = forecasting.get("LOAD_OR_COMPUTE_CV")
            if forecasting.get("SAVE_CV") is not None:
                self.save_cv = forecasting.get("SAVE_CV")
