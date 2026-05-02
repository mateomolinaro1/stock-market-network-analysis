"""
Standalone entry point for the oracle (perfect-foresight) timing backtest.
The logic lives in stock_mkt_network_analysis.experiments.timing_backtest.
"""
from dotenv import load_dotenv
import logging
import sys

from stock_mkt_network_analysis.utils.config import Config
from stock_mkt_network_analysis.data.data_manager import DataManager
from stock_mkt_network_analysis.experiments.timing_backtest import run_oracle_timing_backtest

if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = Config()
    data_manager = DataManager(config=config)
    data_manager.load_data()
    output_dir = config.ROOT_DIR / "outputs" / "figures" / "timing_backtest"
    run_oracle_timing_backtest(config, data_manager, output_dir)