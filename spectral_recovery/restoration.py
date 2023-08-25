import xarray as xr
import geopandas as gpd
from pandas import Index

from typing import Callable, Optional, Union, List
from datetime import datetime
from pandas import Timestamp
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

    """

    def __init__(
        self,
        reference_polygons: gpd.GeoDataFrame,
        reference_stack: xr.DataArray,
        reference_range: Union[int, List[int]],
        baseline_method: Optional[Callable] = None,
    ) -> None:
        # TODO: convert date inputs into standard form (pd.dt?)
        self.reference_polygons = reference_polygons
        self.reference_range = to_datetime(reference_range)
        self.baseline_method = baseline_method or historic_average
        if not self._within(reference_stack):
            raise ValueError(f"Not contained! Better message soon!")
        else:
            clipped_stacks = {}
            # TODO: Maybe handle MultiPolygons here. Otherwise force everything to Polygon in pre-processing.
            for i, row in reference_polygons.iterrows():
                polygon_stack = reference_stack.rio.clip(gpd.GeoSeries(row.geometry).values)
                clipped_stacks[i] = polygon_stack
            self.reference_stack = xr.concat(clipped_stacks.values(), dim=Index(clipped_stacks.keys(), name="poly_id"))
            print(self.reference_stack)

    def baseline(self):
        # TODO: replace return dicts with named tuple
        """Get the baseline/recovery target for a reference system"""
        baseline = self.baseline_method(self.reference_stack, self.reference_range)
        return {"baseline": baseline}

    def _within(self, stack: xr.DataArray) -> bool:
        """Check if within a DataArray

        Determines whether an RestorationArea's spatial (polygons) and temporal
        (reference and event years) attributes are contained within a
        stack of yearly composite images.

        """
        if not (
            stack.yearcomp.contains_spatial(self.reference_polygons)
            and stack.yearcomp.contains_temporal(self.reference_range)
        ):
            return False
        return True


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

        try:
            _ = len(restoration_year)
            raise TypeError(
                "Iterable passed to restoration_year, but restoration_year must be a Timestamp."
            )
        except:
            if type(restoration_year) is Timestamp:
                self.restoration_year = restoration_year
            else:
                self.restoration_year = to_datetime(restoration_year)

        if not isinstance(reference_system, ReferenceSystem):
            historic_reference = ReferenceSystem(
                reference_polygons=restoration_polygon,
                reference_stack=composite_stack,
                reference_range=reference_system,
            )
            self.reference_system = historic_reference
        else:
            self.reference_system = reference_system

        if not self._within(composite_stack):
            raise ValueError(f"Not contained! Better message soon!")
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

    def metrics(self, metrics: List[str]) -> xr.DataArray:
        """Generate recovery metrics over a Restoration Area

        Parameters
        ----------
        metrics : list of Index
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
            match metric:
                case Metric.percent_recovered:
                    curr = self.stack.sel(time=self.end_year)
                    baseline = self.reference_system.baseline()
                    event = self.stack.sel(time=self.restoration_year)
                    metrics_dict[metric] = percent_recovered(
                        eval_stack=curr, baseline=baseline["baseline"], event_obs=event
                    )
                case Metric.years_to_recovery:
                    filtered_stack = self.stack.sel(
                        time=slice(self.restoration_year, self.end_year)
                    )
                    baseline = self.reference_system.baseline()
                    metrics_dict[metric] = years_to_recovery(
                        image_stack=filtered_stack,
                        baseline=baseline["baseline"],
                    )
                case Metric.dNBR:
                    metrics_dict[metric] = dNBR(
                        restoration_stack=self.stack,
                        rest_start=str(self.restoration_year.year),
                    )
                case Metric.recovery_indicator:
                    metrics_dict[metric] = recovery_indicator(
                        image_stack=self.stack,
                        rest_start=str(self.restoration_year.year),
                    )
        metrics_stack = _stack_bands(
            metrics_dict.values(), metrics_dict.keys(), dim_name="metric"
        )
        return metrics_stack
