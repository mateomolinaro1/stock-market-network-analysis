import pandas as pd
from stock_mkt_network_analysis.utils.config import Config
from stock_mkt_network_analysis.data.data_manager import DataManager
from dotenv import load_dotenv
load_dotenv()
config = Config()

data_manager = DataManager(config=config)
data_manager.load_data()