from Context import Context
from typing import Tuple, Union, Optional
import pandas as pd
import logging
from dataclasses import dataclass
from Metadata import Metadata

""" This is to allow users fleibility in choosing the split location in the
dataset.  Option parsing like this is verbose, therefore this code has been
isolated in its own file.
"""


def parse_split_setting(
    ctx: Context, md: Metadata, df: pd.DataFrame
) -> Tuple[str, Optional[pd.Timestamp]]:
# ):
    """ Parses the user-supplied split option.

    Available split options are:
        * split_date
            check type (Timestamp)
            check within range (oldest .. newest)
            split_date = used directly
        * split_fraction
            check type (float)
            check within range (0.0 .. 1.0)
            split_date = date of row[size x (1.0 - split_fraction)]
        * split_test_timedelta
            check type (Timedelta)
            check within range (total timedelta of dataset)
            split_date = newest data - split_test_timedelta

    Args:
        * ctx: Context dataclass, containing case-specific parameters
        * md: Metadata dataclass, containing properties of the dataset
        * df: pd.Dataframe, the raw dataset

    One or None of these are allowed.  If two or more are specified, a warning
    is issued, and the last one parsed gets used.

    These are all user-facing options, but internally, they are calculated and
    converted to the split_date type

    Returns:
        (split_type, split_date)
    """
    split_type = ""
    split_date = None
    n_split_vars = 0
    if ctx.split_date is not None:
        if not isinstance(ctx.split_date, pd.Timestamp):
            msg = (
                f"split_date should be type 'pandas.Timestamp', got "
                f"'{type(ctx.split_date)}'"
            )
            logging.error(msg)
            raise ValueError(msg)

        t_old = md.date_oldest
        t_new = md.date_newest
        if not (t_old <= ctx.split_date < t_new):
            msg = (
                f"Invalid split_date: {ctx.split_date} is outside the data"
                f" range.\n"
                f"\tData starts at {t_old}, ends at {t_new}."
            )
            logging.error(msg)
            raise ValueError(msg)
        split_type = "Timestamp"
        split_date = ctx.split_date
        n_split_vars += 1

    if ctx.split_fraction is not None:
        if not isinstance(ctx.split_fraction, float):
            msg = (
                f"split_fraction should be type 'float', got "
                f"'{type(ctx.split_fraction)}'"
            )
            logging.error(msg)
            raise ValueError(msg)
        if not (0.0 <= ctx.split_fraction < 1.0):
            msg = (
                f"Invalid split_fraction: {ctx.split_fraction} is outside the "
                f" expected range.\n"
                f"\texpecting a value between 0.0 and 1.0."
            )
            logging.error(msg)
            raise ValueError(msg)
        n_test = int(md.count*ctx.split_fraction)
        test_starts_at = md.count - n_test - 1
        if test_starts_at > md.count or test_starts_at < 0:
            msg = (
                f"Failed to split data set into training / test segments.\n"
                f"\tn_test         : {n_test},\n"
                f"\ttest_starts_at : {test_starts_at},\n"
                f"\ttotal size     : {md.count}"
            )
            logging.error(msg)
            raise ValueError(msg)
        split_date = df['Timestamp'].iloc[test_starts_at]
        split_type = "fraction"
        n_split_vars += 1

    if ctx.split_test_timedelta is not None:
        if not isinstance(ctx.split_test_timedelta, pd.Timedelta):
            msg = (
                f"split_test_timedelta should be type 'pandas.Timedelta', "
                f"got '{type(ctx.split_test_timedelta)}'"
            )
            logging.error(msg)
            raise ValueError(msg)
        if ctx.split_test_timedelta >= md.date_timedelta:
            msg = (
                f"Invalid split_test_timedelta: "
                f"{ctx.split_test_timedelta} \n"
                f"\tLonger than the full dataset: "
                f"{md.date_timedelta}."
            )
            logging.error(msg)
            raise ValueError(msg)
        split_date = md.date_newest - ctx.split_test_timedelta
        split_type = "test_timedelta"
        n_split_vars += 1

    if n_split_vars == 0:
        logging.info(
            "No time split specified, using full data for training"
        )
        split_type = ""
        split_date = None
    elif n_split_vars > 1:
        logging.warning(
            "More than one split option found.\n"
            f"\tsplit_date: {ctx.split_date}\n"
            f"\tsplit_fraction: {ctx.split_fraction}\n"
            f"\tsplit_test_timedelta: {ctx.split_test_timedelta}\n"
            f"using '{split_type}'"
        )

    return split_type, split_date
