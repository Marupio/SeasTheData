from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import pandas as pd

@dataclass
class Context:
    """Contains context about current run conditions."""

    # Input filepath
    path_in: Path

    # Dataset split into 'training' and 'testing', all data before split_date
    # are 'training'.  If all of these are 'None', all data are training.

    # split_date - exact timestamp for split
    split_date: Optional[pd.Timestamp]=None

    # split_fraction - fraction of datapoints that are test (most recent)
    split_fraction: Optional[float]=None

    # split_test_timedelta: duration of test period (most recent)
    split_test_timedelta: Optional[pd.Timedelta]=None

    # When true, plots interim results
    plot: bool=True

    # Logging mode
    verbose: bool=False
    debug: bool=False
