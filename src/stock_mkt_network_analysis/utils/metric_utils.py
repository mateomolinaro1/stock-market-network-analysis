"""
This module provides utility functions for calculating various metrics related to stock market network analysis.
"""

from typing import Callable
import pandas as pd
from beartype import beartype

class Metrics:
    def __init__(self):
        pass

    @staticmethod
    def compute_realized_volatility(df: pd.DataFrame, rolling_window: int, data_freq: str = 'd') -> pd.DataFrame:
        """
        Compute the annualized realized volatility from the market returns.
        :return: a dataframe with dates as index and a single column 'annualized_realized_volatility' as values.
        """
        mapping_freq_to_annualization_factor = {
            'd': 252,
            'w': 52,
            'm': 12,
            'y': 1
        }
        annualization_factor = mapping_freq_to_annualization_factor.get(data_freq)
        realized_volatility = df.rolling(window=rolling_window).std() * (annualization_factor ** 0.5)
        realized_volatility.rename(columns={realized_volatility.columns[0]: 'annualized_realized_volatility'+f'_{realized_volatility.columns[0]}'}, inplace=True)
        return realized_volatility

    @staticmethod
    def compute_maximum_drawdown(
            df: pd.DataFrame,
            rolling_window: int
    ) -> pd.DataFrame:
        """
        Compute the maximum drawdown for the df provided.
        :return: a dataframe with dates as index and a single column 'maximum_drawdown' as values.
        """
        cumulative_returns = (1 + df).rolling(window=rolling_window).apply(lambda x: x.prod())
        running_max = cumulative_returns.cummax()
        drawdown = cumulative_returns / running_max - 1
        drawdown.rename(columns={drawdown.columns[0]: 'maximum_drawdown'+f'_{drawdown.columns[0]}'}, inplace=True)
        return drawdown

    @staticmethod
    def compute_dummy_from_feature(
            df: pd.DataFrame,
            rolling_window: int,
            feature_func: Callable[[pd.DataFrame, int], pd.DataFrame],
            quantile: float = 0.5,
            output_col_name: str = 'dummy_feature'
    ) -> pd.DataFrame:
        feature = feature_func(df, rolling_window)

        rolling_quantile = feature.rolling(window=rolling_window).quantile(quantile)
        dummy_feature = (feature < rolling_quantile).astype(int)
        dummy_feature.rename(columns={dummy_feature.columns[0]: output_col_name+f'_{dummy_feature.columns[0]}'}, inplace=True)
        return dummy_feature

    @staticmethod
    def compute_cumulative_return(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the cumulative return from a df returns.
        :return: a dataframe with dates as index and a single column 'cumulative_return' as values.
        """
        cumulative_return = (1 + df).cumprod() - 1
        cumulative_return.rename(columns={cumulative_return.columns[0]: 'cumulative_return'+f'_{cumulative_return.columns[0]}'}, inplace=True)
        return cumulative_return