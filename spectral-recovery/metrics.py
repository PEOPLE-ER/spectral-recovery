import xarray as xr

from enum import Enum
from utils import maintain_spatial_attrs

class Metrics(Enum):
    percent_recovered = "percent_recovered"
    years_to_recovery = "Y2R"

    def __str__(self) -> str:
        return self.name

@maintain_spatial_attrs
def percent_recovered(
        stack: xr.DataArray,
        baseline: xr.DataArray,
        event: xr.DataArray
    ) -> xr.DataArray:
    total_change = abs(baseline-event)
    recovered = abs(stack-baseline)
    return recovered / total_change
