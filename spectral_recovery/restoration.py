import xarray as xr
import geopandas as gpd
from pandas import Index

from typing import Callable, Optional, Union, List
from datetime import datetime
from pandas import Timestamp
from spectral_recovery.timeseries import _stack_bands
from spectral_recovery.recovery_target import historic_average
from spectral_recovery.utils import to_datetime
from spectral_recovery.metrics import (
    percent_recovered,
    years_to_recovery,
    dNBR,
    recovery_indicator,
)
from spectral_recovery.enums import Metric

# TODO: split date into start and end dates.
# TODO: remove baseline_method as attribute. Add it as a parameter to baseline()
class ReferenceSystem:

    """A Reference System.

    Attributes
    -----------
    reference_polygons : gpd.GeoDataframe
        The spatial deliniation of the reference areas. GeoDataframe
        must contain at least one geometry and must be
        of type shapely.Polygon or shapely.MultiPolygon
    reference_stack: xr.DataArray
        A 4D (band, time, y, x) DataArray of images from which indices and
        metrics will be computed. The spatial bounds of the DataArray must 
        contain `restoration_polygon` and (optional) `reference_polygons`, 
        and the temporal bounds must contain `restoration_year`.
    reference_years : Tuple of datetimes
        The year or range of years from which to get values for computing
        the recovery target.
    recovery_target_method : Callable
        The method for computing the recovery target value. Must operate on
        4D (band, time, y, x) DataArrays.

    """
    def __init__(
        self,
        reference_polygons: gpd.GeoDataFrame,
        reference_stack: xr.DataArray,
        reference_range: Union[int, List[int]],
        recovery_target_method: Optional[Callable] = None,
    ) -> None:
        # TODO: convert date inputs into standard form (pd.dt?)
        self.reference_polygons = reference_polygons
        self.reference_range = to_datetime(reference_range)
        self.recovery_target_method = recovery_target_method or historic_average
        if not self._within(reference_stack):
            raise ValueError(f"Not contained! Better message soon!")
        else:
            clipped_stacks = {}
            # TODO: Maybe handle MultiPolygons here. Otherwise force everything to Polygon in pre-processing.
            for i, row in reference_polygons.iterrows():
                polygon_stack = reference_stack.rio.clip(gpd.GeoSeries(row.geometry).values)
                clipped_stacks[i] = polygon_stack
            self.reference_stack = xr.concat(clipped_stacks.values(), dim=Index(clipped_stacks.keys(), name="poly_id"))

    def recovery_target(self):
        # TODO: replace return dicts with named tuple
        """Get the recovery target for a reference system"""
        recovery_target = self.recovery_target_method(self.reference_stack, self.reference_range)
        return {"recovery_target": recovery_target}

    def _within(self, stack: xr.DataArray) -> bool:
        """Check if within a DataArray

        Determines whether an RestorationArea's spatial (polygons) and temporal
        (reference and event years) attributes are contained within a
        stack of yearly composite images.

        """
        if not (
            stack.satts.contains_spatial(self.reference_polygons)
            and stack.satts.contains_temporal(self.reference_range)
        ):
            return False
        return True

# TODO: split reference_system param to reference_polygon and reference_date params
# and then allow the reference polygons to be passed directly to RestorationArea.
# This will pass make RestorationArea the sole object responsible for instatiated a ReferenceSystem
class RestorationArea:
    """A Restoration Area (RA).

    Attributes
    -----------
    restoration_polygon : GeoDataFrame
        The spatial deliniation of the restoration event. There 
        must only be one geometry in the GeoDataframe and it must be
        of type shapely.Polygon or shapely.MultiPolygon.
    restoration_year : str or datetime
        The start year of the restoration event. Value must be within 
        the time dimension coordinates of `composite_stack` param.
    reference_polygons : int or list of int or ReferenceSystem
        The reference system for the restoration area. If ints, then
        a ref sys will created as a Historic Reference System and 
        will reference the dates indicated by ints as the reference times. 
    composite_stack : xr.DataArray
        A 4D (band, time, y, x) DataArray of images from which indices and
        metrics will be computed. The spatial bounds of the DataArray must 
        contain `restoration_polygon` and (optional) `reference_polygons`, 
        and the temporal bounds must contain `restoration_year`.

    """
    def __init__(
        self,
        restoration_polygon: gpd.GeoDataFrame,
        restoration_year: str | datetime,
        reference_system: int | List[int] | ReferenceSystem,
        composite_stack: xr.DataArray,
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
        self.end_year = self.stack["time"].max()

    def _within(self, stack: xr.DataArray) -> bool:
        """Check if within a DataArray

        Determines whether an RestorationArea's spatial (polygons) and temporal
        (reference and event years) attributes are contained within a
        stack of yearly composite images.

        """
        if not (
            stack.satts.contains_spatial(self.restoration_polygon)
            and stack.satts.contains_temporal(self.restoration_year)
            and stack.satts.contains_temporal(self.reference_system.reference_range)
        ):
            return False
        return True

    def metrics(self, metrics: List[str]) -> xr.DataArray:
        """Generate recovery metrics over a Restoration Area

        Parameters
        ----------
        metrics : list of str
            A list containing the names of metrics to generate.

        Returns
        -------
        metrics_stack : xr.DataArray
            A 3D (metric, y, x) DataArray, computed metrics values are 
            located along the `metric` dimension. Coordinates of metrics
            are the related enums.Metric.

        """
        metrics_dict = {}
        for metrics_input in metrics:
            metric = Metric(metrics_input)
            match metric:
                case Metric.percent_recovered:
                    curr = self.stack.sel(time=self.end_year)
                    recovery_target = self.reference_system.recovery_target()
                    event = self.stack.sel(time=self.restoration_year)
                    metrics_dict[metric] = percent_recovered(
                        image_stack=curr, recovery_target=recovery_target["recovery_target"], event_obs=event
                    )
                case Metric.years_to_recovery:
                    filtered_stack = self.stack.sel(
                        time=slice(self.restoration_year, self.end_year)
                    )
                    recovery_target = self.reference_system.recovery_target()
                    metrics_dict[metric] = years_to_recovery(
                        image_stack=filtered_stack,
                        recovery_target=recovery_target["recovery_target"],
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
