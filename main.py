from src.stock_mkt_network_analysis.utils.config import Config
from src.stock_mkt_network_analysis.data.data_manager import DataManager
from src.stock_mkt_network_analysis.analytics.analytics import Analytics
from dotenv import load_dotenv
import logging
import sys

# Config
load_dotenv()
config = Config()
logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Data
data_manager = DataManager(config=config)
data_manager.load_data()

# Analytics
an = Analytics(config=config, data=data_manager)
an.get_analytics()