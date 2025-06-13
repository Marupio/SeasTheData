""" This is the main dataset forecast engine, written from scratch
See also:
    ParseSplit.py for some user option parsing, moved out-of-file in order
        to keep the data science elements clean.
    Context.py a dataclass to share runtime context with different apps / use
        cases
    Metadata.py a simple dataclass to hold select metadata about the dataset
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from Model import RedemptionModel
from Context import Context
from Metadata import Metadata
from ParseSplit import parse_split_setting

# Global functions
def set_logging(ctx):
    """ Engage logging based on CLI context
    """
    if ctx.debug:
        log_level = logging.DEBUG
    elif ctx.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=log_level,
            format="%(levelname)s: %(message)s"
        )
    else:
        root_logger.setLevel(log_level)


class AnalysisEngine:

    ### STATIC MEMBER DATA ###

    # Expected headings in the raw CSV file
    raw_data_headings = ['_id', 'Timestamp', 'Redemption Count', 'Sales Count']


    ### MEMBER FUNCTIONS ARE SORTED BY CHRONOLOGICAL SEQUENCE (where possible)

    def __init__(self, ctx: Context):
        """ Construct this class, requires the 'context' """
        self.ctx = ctx


    # API - dev-facing function
    def read_csv(self, path: Path):
        """ Read the expected csv file from the supplied path
        Args:
            path - path to file with csv data

        Data gets tested for validity, such as having all expected keys present:
            '_id', 'Timestamp', 'Redemption Count', 'Sales Count'

        Succeeds silently.  Raises exceptions on failure.
        """
        if hasattr(self, "df"):
            error_str = (
                f"Attempting to read csv when data already exists.\n"
                f"\tPath: {path}"
            )
            logging.error(error_str)
            raise RuntimeError(error_str)
        logging.info(f"Reading data from {path}")
        self.df_raw = pd.read_csv(
            path,
            dtype={'_id':int, 'Redemption Count': int, 'Sales Count':int},
            parse_dates=['Timestamp']
        )
        # Ensure data is non-empty and contains the expected headings
        self._validate()

        # Split dataset into training and testing subsets
        self._split_data()


    def _validate(self) -> None:
        """ Checks csv data that was just loaded into df_raw.  Raises an
        exception if data is not valid or is empty.
        """
        if not isinstance(self.df_raw, pd.DataFrame):
            raise TypeError(
                f"Expecting df to be DataFrame, got {type(self.df_raw)}"
            )
        if self.df_raw.empty:
            raise pd.errors.EmptyDataError("df contains no data")

        keys = self.df_raw.keys()
        if not all(rdh in keys for rdh in self.raw_data_headings):
            missing_keys = (
                [rdh for rdh in self.raw_data_headings if rdh not in keys]
            )
            raise KeyError(
                f"Some expected keys missing:\n"
                f"\tMissing keys: {missing_keys}\n"
                f"\tAll found keys: {keys}"
            )

        # Parse Timestamp column
        self.df_raw['Timestamp'] = pd.to_datetime(self.df_raw['Timestamp'])
        # Sort oldest first (df_raw.index[0]), newest last
        self.df_raw.sort_values('Timestamp', inplace=True)

        # Get metadata - temporal domain, dataset size
        self._get_metadata()

        # Check 'user' inputs
        self._check_ctx()


    def _get_metadata(self):
        """ Grab metadata from the dataset and store it in the Metadata
        dataclass
        """
        date_oldest = self.df_raw['Timestamp'].iloc[0]
        date_newest = self.df_raw['Timestamp'].iloc[-1]
        assert date_oldest < date_newest, "Data is not sorted correctly."
        date_timedelta = date_newest - date_oldest
        count = len(self.df_raw)
        self.metadata = Metadata(
            date_oldest=date_oldest,
            date_newest=date_newest,
            date_timedelta=date_timedelta,
            count=count
        )


    def _check_ctx(self):
        """ Ensure context settings make sense and have correct types
        """
        # Check Context, self.ctx
        c = self.ctx # brevity

        if not isinstance(c.path_in, Path):
            msg = (
                f"path_in should be type 'Path', got '{type(c.path_in)}'"
            )
            logging.error(msg)
            raise TypeError(msg)

        # Check data split settings - parsing moved to keep this class clean
        self.split_type, self.split_date = parse_split_setting(
            self.ctx, self.metadata, self.df_raw
        )

        if not isinstance(c.plot, bool):
            msg = (
                f"plot should be type 'bool', got '{type(c.plot)}'"
            )
            logging.error(msg)
            raise TypeError(msg)

        if not isinstance(c.debug, bool):
            msg = (
                f"debug should be type 'bool', got '{type(c.debug)}'"
            )
            logging.error(msg)
            raise TypeError(msg)

        if not isinstance(c.verbose, bool):
            msg = (
                f"verbose should be type 'bool', got '{type(c.verbose)}'"
            )
            logging.error(msg)
            raise TypeError(msg)


    def _split_data(self):
        """ Split the dataset into training and test subsets.
        Users can select the split location using any of three methods.  All
        get converted into the equivalent value as split_date.
        I should have used TimeSeriesSplit
        """
        sd = self.split_date
        if sd is None:
            self.df_train = self.df_raw.copy()
            self.df_test = pd.DataFrame([])
            logging.info("Using full data set for training")
        else:
            self.df_train = self.df_raw[self.df_raw['Timestamp'] < sd].copy()
            self.df_test = self.df_raw[self.df_raw['Timestamp'] >= sd].copy()
            logging.info(
                f"Training set : {len(self.df_train)}\n"
                f"Testing set  : {len(self.df_test)}"
            )


    def prepare_for_projection(self):
        """ Performs all operations and calculations necessary prior to calling
        any projection functionality
        """
        self._create_sub_day_profile()
        self._create_weekly_profile()
        self._create_annual_profile()
        self._create_overall_trend()
        assert hasattr(self, 'df_yearly'), "df_yearly not available"
        assert hasattr(self, 'df_yearly_rolling'), (
            "df_yearly_rolling not available"
        )
        assert hasattr(self, 'profile_annual'), "profile_annual not available"
        assert hasattr(self, 'profile_weekly'), "profile_weekly not available"


    def _create_sub_day_profile(self):
        """ Create a profile for a typical day.

        Profile properties:
            * Total duration = 1 day
            * Point spacing  = 15 min
        """
        logging.info("Component 1 - sub-day profile")

        # Extract time-of-day component (ignoring date)
        self.df_train['Time_of_day'] = (
            self.df_train['Timestamp'].dt.strftime('%H:%M')
        )

        # Group by Time_of_day and compute statistics
        self.profile_sub_day = (
            self.df_train.groupby(
                'Time_of_day'
            )[['Redemption Count', 'Sales Count']]
                .agg(['mean', 'std', 'count'])
        )
        self.profile_sub_day.columns = (
            [
                '_'.join(col).strip()
                for col in self.profile_sub_day.columns.values
            ]
        )
        # Create a normalised profile
        self.pn_sub_day = (
            self.profile_sub_day[['Redemption Count_mean', 'Sales Count_mean']]
        )
        self.pn_sub_day = self.pn_sub_day / self.pn_sub_day.mean()


    def _create_weekly_profile(self):
        """ Create a profile for a typical week.

        Profile properties:
            * Total duration = 7 days
            * Point spacing  = 1 day
        """
        logging.info("Component 2 - weekly profile")

        # Roll up the 15-minute entries into one row per day
        self.df_daily = (
            self.df_train
            .resample('D', on='Timestamp')[['Redemption Count', 'Sales Count']]
            .sum()
        )
        self.df_daily = self.df_daily.reset_index()

        # Extract the day of the week
        # This yields integers from 0 (Monday) to 6 (Sunday)
        self.df_daily['Day_of_week'] = self.df_daily['Timestamp'].dt.dayofweek

        # Group by day and compute average totals
        self.profile_weekly = (
            self.df_daily
            .groupby('Day_of_week')[['Redemption Count', 'Sales Count']].mean()
        )

        # Create a normalised profile
        self.pn_weekly = self.profile_weekly / self.profile_weekly.mean()


    def _create_annual_profile(self):
        """ Create a profile for a typical year.

        Profile properties:
            * Total duration = 52 weeks
            * Point spacing  = 1 week
        """
        logging.info("Component 3 - annual profile")

        # Working with Timestamp as index
        self.df_train = self.df_train.set_index('Timestamp')

        # Resample to weekly totals
        self.df_weekly = (
            self.df_train
            .resample('W')[['Redemption Count', 'Sales Count']].sum()
        )
        self.df_weekly = self.df_weekly.reset_index()

        # Identify week index within a year, 1..52
        self.df_weekly['WeekOfYear'] = (
            self.df_weekly['Timestamp'].dt.isocalendar().week
        )

        # Average by week number, (small sample size at this time scale)
        self.profile_annual = (
            self.df_weekly.groupby(
                'WeekOfYear'
            )[['Redemption Count', 'Sales Count']].mean()
        )

        # Create a normalised profile
        self.pn_annual = self.profile_annual / self.profile_annual.mean()


    def _create_overall_trend(self):
        """ Create trend data for the entire dataset

        Uses three different methods:

        * Take1 - Rolling average: uses a 30-day centered window
            Not a successful approach
        * Take2 - Aggregates entire years into single data points
            Similar to profile calculations.  Noisy.
        * Take3 - Same as take2, but smoothed out with 3-year centered windows
        """
        logging.info("Component 4 - overall trend")
        # Take1 - Rolling average

        # Daily totals
        self.df_daily = (
            self.df_train
            .resample('D')[['Redemption Count', 'Sales Count']]
            .sum()
        )

        # Rolling trend (30-day centered window)
        self.trend_rolling = (
            self.df_daily.rolling(window=30, center=True).mean()
        )

        # Take2 - entire year summations
        # Add a year column
        dt_index = self.df_train.index
        self.df_train['Year'] = (
            dt_index.year # type: ignore[reportAttributeAccessIssue]
        )

        # Group by year and sum Sales and Redemptions
        self.df_yearly = (
            self.df_train
            .groupby('Year')[['Redemption Count', 'Sales Count']]
            .sum()
        )

        # Take3 - entire year, with rolling window
        self.df_yearly_rolling = (
            self.df_yearly.rolling(window=3, center=True).mean().dropna()
        )


    def extrapolate_year(self, target_year: int):
        """Fit linear regression to df_yearly_rolling and extrapolate."""
        logging.info(f"Extrapolating to year {target_year}")
        # Always work with the original data - restore if exists, backup
        if hasattr(self, 'df_yearly_rolling_orig'):
            self.df_yearly_rolling = self.df_yearly_rolling_orig
        else:
            self.df_yearly_rolling_orig = self.df_yearly_rolling.copy()
        oldest_year = self.df_yearly_rolling.index[0]
        newest_year = self.df_yearly_rolling.index[-1]
        assert oldest_year < newest_year, (
            f"yearly data not sorted correctly:\n"
            f"\toldest={oldest_year}\n"
            f"\tnewest={newest_year}"
        )
        if target_year < oldest_year:
            msg = (
                f"Attempting to project into the past:\n"
                f"\ttarget year ={target_year}\n"
                f"\toldest year ={oldest_year}\n"
                f"\tnewest year ={newest_year}"
            )
            logging.error(msg)
            raise ValueError(msg)
        if target_year < newest_year:
            logging.warning(
                f"Extrapolate called when unnecessary:\n"
                f"\toldest year ={oldest_year}\n"
                f"\ttarget year ={target_year}\n"
                f"\tnewest year ={newest_year}"
            )
            return
        n_years = target_year - newest_year
        df = self.df_yearly_rolling

        # Change array dimensions for scikit
        X = df.index.values.reshape(-1, 1)  # e.g., [[2017], [2018], ...]
        projections = {}

        for col in df.columns:
            # y = df[col].values
            y = np.asarray(df[col].values)  # Ensures it's a plain ndarray
            model = LinearRegression().fit(X, y)

            future_years = (
                np.arange(X[-1][0] + 1, X[-1][0] + n_years + 1).reshape(-1, 1)
            )
            y_future = model.predict(future_years)
            projections[col] = y_future

        # Assemble projected DataFrame
        df_future = (
            pd.DataFrame(
                projections,
                index=future_years.flatten()  # type: ignore[arg-type]
            )
        )
        df_future.index.name = 'Year'

        self.df_yearly_rolling = pd.concat([df, df_future])


    def project_sub_day_interval(self, target_time: pd.Timestamp):
        """ Create a prediction for the 15 minute interval at the given
        timestamp.
        """
        # Get day, time, week-of-year, and year
        time_of_day = target_time.strftime('%H:%M')
        day_of_week = target_time.dayofweek
        week_of_year = target_time.isocalendar().week
        year = target_time.year

        # Fetch normalised scale factors
        scale_sub_day = self.pn_sub_day.loc[time_of_day]
        scale_weekly = self.pn_weekly.loc[day_of_week]
        scale_annual = self.pn_annual.loc[week_of_year]
        if not year in self.df_yearly_rolling.index:
            self.extrapolate_year(year)
        base_year = self.df_yearly_rolling.loc[year]

        # Estimate number of intervals per year
        intervals_per_year = 365 * 24 * 4
        base_redemption = base_year['Redemption Count'] / intervals_per_year
        base_sales = base_year['Sales Count'] / intervals_per_year

        # Combine components
        redemptions = (
            base_redemption *
            scale_annual['Redemption Count'] *
            scale_weekly['Redemption Count'] *
            scale_sub_day['Redemption Count_mean']
        )

        sales = (
            base_sales *
            scale_annual['Sales Count'] *
            scale_weekly['Sales Count'] *
            scale_sub_day['Sales Count_mean']
        )
        return redemptions, sales


    def project_day_total(self, date: pd.Timestamp):
        """ Create a prediction for a full day of sales and redemptions at the
        given timestamp.
        """
        redemptions_total, sales_total = 0.0, 0.0
        for minute in range(0, 24*60, 15):
            timestamp = pd.Timestamp(date) + pd.Timedelta(minutes=minute)
            r, s = self.project_sub_day_interval(timestamp)
            redemptions_total += r
            sales_total += s
        return redemptions_total, sales_total


    def project_week_total(self, start_date: pd.Timestamp):
        """ Create a prediction for a full week of sales and redemptions at the
        given timestamp.
        """
        redemptions_total, sales_total = 0.0, 0.0
        for d in range(7):
            day = start_date + pd.Timedelta(days=d)
            r, s = self.project_day_total(day)
            redemptions_total += r
            sales_total += s
        return redemptions_total, sales_total


    def create_full_year_projection(self, year: int) -> pd.DataFrame:
        """
        Generate a full year of sub-day (15-minute interval) projections
        for the specified year using the AnalysisEngine instance.

        Args:
            year (int): Year to project.

        Returns:
            pd.DataFrame: DataFrame with timestamps as index and two columns:
                        'Redemption Count' and 'Sales Count'.
        """
        logging.info(f"Generating full-year projection for {year}")

        # Generate all 15-minute timestamps in the year
        start = pd.Timestamp(datetime(year, 1, 1))
        end = pd.Timestamp(datetime(year + 1, 1, 1))
        timestamps = pd.date_range(start=start, end=end, freq='15min')[:-1]

        # Generate projections for each timestamp
        projections = [self.project_sub_day_interval(ts) for ts in timestamps]

        # Build DataFrame
        df_projection = pd.DataFrame(
            projections,
            index=timestamps,
            columns=['Redemption Count', 'Sales Count']
        )

        logging.info(
            f"Projection complete: {len(df_projection)} intervals generated."
        )
        return df_projection


    def _prepare_test_dataset(self):
        """ Add any columns to the test dataset that have been added to the
        training dataset
        """
        if hasattr(self, 'df_test_ready'):
            logging.warning("Attempting to prepare test data more than once")
            return
        # Prepare testing dataset - add columns that exist in training set
        self.df_test['Time_of_day'] = (
            self.df_test['Timestamp'].dt.strftime('%H:%M')
        )
        self.df_test = self.df_test.set_index('Timestamp')
        dt_index = self.df_test.index
        self.df_test['Year'] = (
            dt_index.year # type: ignore[reportAttributeAccessIssue]
        )
        self.df_test_ready = True


    def evaluate_uncertainty(self):
        """
        Evaluate model uncertainty using the test dataset (df_test).
        Computes residuals, error metrics (MAE, RMSE, STD), and stores:
            - df_test_projections: projections for test set
            - df_residuals: projection - actual
            - uncertainty_metrics: DataFrame with MAE, RMSE, STD by column
        """
        self._prepare_test_dataset()
        assert hasattr(self, 'df_test'), "df_test not prepared"
        assert hasattr(self, 'df_train'), "df_train not available"

        logging.info("Evaluating uncertainty using test set")

        # Generate predictions for each timestamp in test set
        timestamps = self.df_test.index
        projections = [
            self.project_sub_day_interval(ts) for ts in timestamps
        ]

        self.df_test_projections = pd.DataFrame(
            projections,
            index=timestamps,
            columns=['Redemption Count', 'Sales Count']
        )

        # Slice actual data
        df_actual = self.df_test[['Redemption Count', 'Sales Count']]

        # Compute residuals
        self.df_residuals = self.df_test_projections - df_actual

        # Compute metrics
        mae = self.df_residuals.abs().mean()
        rmse = (self.df_residuals ** 2).mean().pow(0.5)
        std = self.df_residuals.std()

        # Combine into a DataFrame for easy access
        self.uncertainty_metrics = pd.DataFrame({
            'MAE': mae,
            'RMSE': rmse,
            'STD': std
        })

        logging.info("Uncertainty Evaluation Complete:"
            f"\n\tMAE  :\t{mae}"
            f"\n\tRMSE :\t{rmse}"
            f"\n\tSTD  :\t{std}"
        )


    def add_uncertainty_to_projection(
        self,
        projection: pd.DataFrame,
        uncertainty_type: str
    ):
        """Adds uncertainty bands to a projection DataFrame.

        Args:
            projection (pd.DataFrame): The forecast/projection data.
            uncertainty_type (str): One of ["MAE", "RMSE", "STD"] to determine
                uncertainty width.
        """
        allowed_types = ["MAE", "RMSE", "STD"]
        utype = uncertainty_type.upper()
        if utype not in allowed_types:
            msg = (
                f"Unrecognised uncertainty type: {uncertainty_type}.\n"
                f"\tExpecting: {allowed_types}"
            )
            logging.error(msg)
            raise ValueError(msg)

        if not hasattr(self, "uncertainty_metrics"):
            raise RuntimeError("Uncertainty metrics not found. "
                               "Run evaluate_uncertainty() first.")

        for col in projection.columns:
            delta = self.uncertainty_metrics.loc[col, utype]
            projection[f'{col} +1sigma'] = (
                projection[col] + delta  # type: ignore[reportOperatorIssue]
            )
            projection[f'{col} -1sigma'] = (
                projection[col] - delta   # type: ignore[reportOperatorIssue]
            )
