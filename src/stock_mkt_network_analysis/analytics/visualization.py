"""
This module contains functions for performing analytics on stock market network data.
"""
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class Vizu:
    """
    Visualization class for stock market network analytics.
    """
    def __init__(self):
        pass

    @staticmethod
    def plot_time_series(
            df: pd.DataFrame,
            x_index: bool = True,
            x_col: str = None,
            y_col: str = None,
            y2_col: str = None,
            title: str = None,
            xlabel: str = None,
            ylabel: str = None,
            y2_label: str = None,
            figsize: tuple = (10, 6),
            saving_path: str = None,
            show_plot: bool = False,
            blocking: bool = False,
            date_freq: str = 'M',
    ) -> None:

        fig, ax1 = plt.subplots(figsize=figsize)

        # X-axis
        if x_index:
            x = df.index
        else:
            x = df[x_col]

        # Main series on left axis
        ax1.plot(x, df[y_col], label=y_col)
        ax1.set_ylabel(ylabel)
        ax1.set_xlabel(xlabel)
        ax1.set_title(title)
        ax1.grid(True)

        # Second series on right axis
        ax2 = None
        if y2_col is not None:
            ax2 = ax1.twinx()
            ax2.plot(x, df[y2_col], linestyle='--', color='black', label=y2_col)
            ax2.set_ylabel(y2_label if y2_label is not None else y2_col)

            # make sure only the second axis appears on the right
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()
            ax2.tick_params(axis='y', labelleft=False, left=False)

        # Date formatting
        if isinstance(x, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(x):
            if date_freq == 'M':
                ax1.xaxis.set_major_locator(mdates.MonthLocator())
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            elif date_freq == 'D':
                ax1.xaxis.set_major_locator(mdates.DayLocator())
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            elif date_freq == 'Y':
                ax1.xaxis.set_major_locator(mdates.YearLocator())
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # rotate x labels properly
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            fig.tight_layout()

        # Legend
        lines, labels = ax1.get_legend_handles_labels()
        if ax2 is not None:
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2

        ax1.legend(lines, labels)

        # Save
        if saving_path is not None:
            plt.savefig(saving_path, bbox_inches='tight')
            logger.info(f"Plot saved to {saving_path}")

        # Show
        if show_plot:
            plt.show(block=blocking)
        else:
            plt.close(fig)