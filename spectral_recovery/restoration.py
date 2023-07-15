import xarray as xr
import geopandas as gpd

from typing import Callable, Optional, Union, List
from datetime import datetime
from spectral_recovery.timeseries import _stack_bands
from spectral_recovery.baselines import historic_average
from spectral_recovery.utils import to_datetime
from spectral_recovery.metrics import (
    percent_recovered,
    years_to_recovery,
    dNBR,
    recovery_indicator,
)
from spectral_recovery.enums import Metric


# TODO: change all useage of "baselines" to "recovery target" or "reference target"; baseline and reference are curr used interchangably.
# TODO: split date into start and end dates.
class ReferenceSystem:

    """Encapsulates data and methods related to reference areas.

    Attributes
    -----------
    polygons : gpd.GeoDataframe
        A GeoDataframe containing at least one Polygon. Polygons
        represent areas that are considered to be references.

    reference_years : Tuple of datetimes
        The year or range of years to consider as reference

    baseline_method : Callable
        A function for computing the the reference target value
        within the reference system. Must be able to operate on
        4D (band, time, y, x) DataArrays.

    variation_method : Callable, optional
        The method for reporting/determing reference value variation
        (e.g 2 std). Default is None.
        TODO: this might be replaced with a simple named/str param

    """

    def __init__(
        self,
        reference_polygons: gpd.GeoDataFrame,
        reference_range: Union[int, List[int]],
        baseline_method: Callable = None,
        variation_method: Optional[Callable] = None,
    ) -> None:
        # TODO: convert date inputs into standard form (pd.dt?)
        self.reference_polygons = reference_polygons
        self.reference_range = to_datetime(reference_range)
        self.baseline_method = baseline_method or historic_average
        self.variation_method = variation_method

    def baseline(self, stack):
        # TODO: replace return dicts with named tuple
        """Get the baseline/recovery target for a reference system"""
        baseline = self.baseline_method(stack, self.reference_range)
        if self.variation_method is not None:
            variation = self.variation_method(stack, self.reference_range)
            return {"baseline": baseline, "variation": variation}
        return {"baseline": baseline}


class RestorationArea:
    """Encapsulates data and methods related to restoration areas.

    Attributes
    -----------
    restoration_polygon : GeoDataFrame
        Dataframe containing the spatial deliniation of the
        restoration event. Assumed to be Polygon.
    restoration_year : str or datetime
        The start year of the restoration event.
    reference_system : int or list of int of ReferenceSystem
        The reference system to compute the recovery target value.
        If year or year(s) are provided then a historic reference
        system will be used[TODO], if a ReferenceSystem object is
        provided then target values will be based on reference polygons.
    composite_stack : xr.DataArray
        A 4D (band, time, y, x) DataArray containing spectral or
        index data. The time dimension is expected to be
    end_year : str or datetime
        The final year of the restoration period. If not given, the
        the final timestep along the time dimension of `composite_stack`
        is assumed to be the final year of the restoration period.

    """

    def __init__(
        self,
        restoration_polygon: gpd.GeoDataFrame,
        restoration_year: str | datetime,
        reference_system: int | List[int] | ReferenceSystem,
        composite_stack: xr.DataArray,
        end_year: Optional[str | datetime] = None,
    ) -> None:
        if restoration_polygon.shape[0] != 1:
            raise ValueError(
                f"restoration_polygons contains more than one Polygon."
                f"A RestorationArea instance can only contain one Polygon."
            )

        self.restoration_polygon = restoration_polygon
        self.restoration_year = to_datetime(restoration_year)

        if not isinstance(reference_system, ReferenceSystem):
            historic_reference = ReferenceSystem(
                reference_polygons=restoration_polygon, reference_range=reference_system
            )
            self.reference_system = historic_reference
        else:
            self.reference_system = reference_system

        if not self._within(composite_stack):
            raise ValueError(
                f"RestorationArea not contained by stack. Better message soon!"
            )
        self.stack = composite_stack.rio.clip(self.restoration_polygon.geometry.values)
        if not end_year:
            # TODO: there's a more xarray-enabled way to do this via DatetimeIndex.
            # I know it in my bones. Might not matter though.
            self.end_year = self.stack["time"].max()

    def _within(self, stack: xr.DataArray) -> bool:
        """Check if within a DataArray

        Determines whether an RestorationArea's spatial (polygons) and temporal
        (reference and event years) attributes are contained within a
        stack of yearly composite images.

        """
        if not (
            stack.yearcomp.contains_spatial(self.restoration_polygon)
            and stack.yearcomp.contains_temporal(self.restoration_year)
            and stack.yearcomp.contains_temporal(self.reference_system.reference_range)
        ):
            return False
        return True

    # NOTE: the jury (me) is still out on this implementation of metric calls.
    # The design is not set in stone but it works for now.
    def metrics(self, metrics: List[str]) -> xr.DataArray:
        """Generate recovery metrics over a Restoration Area

        Parameters
        ----------
        metrics : list of str
            A list of metrics to generate.

        Returns
        -------
        metrics_stack : xr.DataArray
            A 3D (metric, y, x) DataArray with metrics
            values stacked along `metric` dimension.

        """
        metrics_dict = {}
        for metrics_input in metrics:
            metric = Metric(metrics_input)
            try:
                metric_func = getattr(self, f"_{metric.name}")
            except Exception:  # TODO: Catch a more specific error than this
                raise ValueError(f"{metric} not implemented")
            metrics_dict[metric] = metric_func()
            metrics_stack = _stack_bands(
                metrics_dict.values(), metrics_dict.keys(), dim_name="metric"
            )
        return metrics_stack

    def _percent_recovered(self) -> xr.DataArray:
        curr = self.stack.sel(time=self.end_year)
        baseline = self.reference_system.baseline(self.stack)
        event = self.stack.sel(time=self.restoration_year)
        return percent_recovered(
            eval_stack=curr, baseline=baseline["baseline"], event_obs=event
        )

    def _years_to_recovery(self) -> xr.DataArray:
        filtered_stack = self.stack.sel(
            time=slice(self.restoration_year, self.end_year)
        )
        baseline = self.reference_system.baseline(self.stack)
        return years_to_recovery(
            image_stack=filtered_stack,
            baseline=baseline["baseline"],
        )

    def _dNBR(self) -> xr.DataArray:
        return dNBR(
            restoration_stack=self.stack, rest_start=str(self.restoration_year.year)
        )

    def _recovery_indicator(self) -> xr.DataArray:
        return recovery_indicator(
            image_stack=self.stack, rest_start=str(self.restoration_year.year)
        )
