import re

DATETIME_FREQ = "YS"

REQ_DIMS = ["band", "time", "y", "x"]

VALID_YEAR = re.compile(r"^\d{4}$")