import pandas as pd 

from datetime import datetime, timezone
from typing import List, Union

def to_datetime(value: Union[str, List[str], datetime, pd.Timestamp]):
    """ Format year and year ranges to UTC datetime."""
    # TODO
    # convert to UTC if not in UTC, if no timezone then assume UTC
    # Convert a year range from X to Y as start-of-year X to end-of-year Y
    return value