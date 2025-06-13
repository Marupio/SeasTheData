from dataclasses import dataclass
import pandas as pd


@dataclass
class Metadata:
    """ Contains metadata about the csv dataset
    """

    # Most recent date appearing in the dataset
    date_newest: pd.Timestamp

    # Oldest date appearing in the dataset
    date_oldest: pd.Timestamp

    # Total timedelta between oldest and newest
    date_timedelta: pd.Timedelta

    # # When True, the dataset is sorted with the oldest entry appearing first
    # first_is_oldest: bool

    # Total number of data points in the dataset
    count: int
