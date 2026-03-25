from dataclasses import dataclass
from pathlib import Path
import logging
import json
from typing import List, Dict, Type, Tuple

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

        # Load JSON config to attributes of Config class
        self._load_run_pipeline_config()

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

