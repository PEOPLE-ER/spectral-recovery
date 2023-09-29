import xarray as xr
import geopandas as gpd
import pandas as pd
from pandas import Index

from typing import Callable, Optional, Union, List
from datetime import datetime
from pandas import Timestamp
from spectral_recovery.timeseries import _stack_bands
from spectral_recovery.recovery_target import historic_average
from spectral_recovery.utils import to_datetime
from spectral_recovery.enums import Metric

from spectral_recovery import metrics as m


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
        and the temporal bounds must contain `restoration_start`.
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
                polygon_stack = reference_stack.rio.clip(
                    gpd.GeoSeries(row.geometry).values
                )
                clipped_stacks[i] = polygon_stack
            self.reference_stack = xr.concat(
                clipped_stacks.values(),
                dim=Index(clipped_stacks.keys(), name="poly_id"),
            )

    def recovery_target(self):
        # TODO: replace return dicts with named tuple
        """Get the recovery target for a reference system"""
        recovery_target = self.recovery_target_method(
            self.reference_stack, self.reference_range
        )
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
# This will make RestorationArea the sole object responsible for instatiating a ReferenceSystem
class RestorationArea:
    """A Restoration Area (RA).

    Attributes
    -----------
    restoration_polygon : GeoDataFrame
        The spatial deliniation of the restoration event. There
        must only be one geometry in the GeoDataframe and it must be
        of type shapely.Polygon or shapely.MultiPolygon.
    disturbance_start : str or datetime
        The year the disturbance began. Value must be within
        the time dimension coordinates of `composite_stack` param.
    restoration_start : str or datetime
        The year the restoration event began. Value must be within
        the time dimension coordinates of `composite_stack` param.
    reference_polygons : int or list of int or ReferenceSystem
        The reference system for the restoration area. If ints, then
        a ref sys will created as a Historic Reference System and
        will reference the dates indicated by ints as the reference times.
    composite_stack : xr.DataArray
        A 4D (band, time, y, x) DataArray of images from which indices and
        metrics will be computed. The spatial bounds of the DataArray must
        contain `restoration_polygon` and (optional) `reference_polygons`,
        and the temporal bounds must contain `restoration_start`.

    """

    def __init__(
        self,
        restoration_polygon: gpd.GeoDataFrame,
        reference_polygon: gpd.GeoDataFrame,
        reference_years: str | List[datetime],
        composite_stack: xr.DataArray,
        disturbance_start: str | datetime = None,
        restoration_start: str | datetime = None,
    ) -> None:
        if restoration_polygon.shape[0] != 1:
            raise ValueError(
                f"restoration_polygons contains more than one Polygon."
                f"A RestorationArea instance can only contain one Polygon."
            )
        self.restoration_polygon = restoration_polygon

        if disturbance_start is None and restoration_start is None:
            return ValueError(
                "At least one of disturbance_start or restoration_start need to be set,"
                " both are None."
            )
        if disturbance_start is not None:
            try:
                _ = len(disturbance_start)
                raise TypeError(
                    "Iterable passed to disturbance_start, but disturbance_start must be"
                    " a Timestamp."
                )
            except:
                if type(disturbance_start) is Timestamp:
                    self.disturbance_start = disturbance_start
                else:
                    self.disturbance_start = to_datetime(disturbance_start)
            if restoration_start is None:
                self.restoration_start = pd.to_datetime(
                    str(self.disturbance_start.year + 1)
                )
                if self.restoration_start < self.disturbance_start:
                    raise ValueError(
                        "The disturbance start year must be less than the restoration start year."
                    )
        if restoration_start is not None:
            try:
                _ = len(restoration_start)
                raise TypeError(
                    "Iterable passed to restoration_start, but restoration_start must"
                    " be a Timestamp."
                )
            except:
                if type(restoration_start) is Timestamp:
                    self.restoration_start = restoration_start
                else:
                    self.restoration_start = to_datetime(restoration_start)
            if disturbance_start is None:
                self.disturbance_start = pd.to_datetime(
                    str(self.restoration_start.year - 1)
                )

            self.reference_system = ReferenceSystem(
                reference_polygons=reference_polygon,
                reference_range=reference_years,
                reference_stack=composite_stack,
                recovery_target_method=None,
            )
        if self.restoration_start < self.disturbance_start:
            # TODO: Should we allow restoration_start=disturbance_start
            raise ValueError(
                "The disturbance start year must be less than the restoration start year."
            )
        if not self._within(composite_stack):
            raise ValueError(f"Not contained! Better message soon!")
        self.stack = composite_stack.rio.clip(self.restoration_polygon.geometry.values)
        self.end_year = pd.to_datetime(self.stack["time"].max().data)

    def _within(self, stack: xr.DataArray) -> bool:
        """Check if within a DataArray

        Determines whether an RestorationArea's spatial (polygons) and temporal
        (reference and event years) attributes are contained within a
        stack of yearly composite images.

        """
        if not (
            stack.satts.contains_spatial(self.restoration_polygon)
            and stack.satts.contains_temporal(self.restoration_start)
            and stack.satts.contains_temporal(self.disturbance_start)
        ):
            return False
        return True

    def Y2R(self, percent_of_target: int = 80):
        post_restoration = self.stack.sel(
            time=slice(self.restoration_start, self.end_year)
        )
        recovery_target = self.reference_system.recovery_target()
        y2r = m.Y2R(
            image_stack=post_restoration,
            recovery_target=recovery_target["recovery_target"],
            rest_start=str(self.restoration_start.year),
            rest_end=str(self.end_year.year),
            percent=percent_of_target,
        )
        y2r = y2r.expand_dims(dim={"metric": [Metric.Y2R]})
        return y2r

    def YrYr(self, timestep: int = 5):
        yryr = m.YrYr(
            image_stack=self.stack,
            rest_start=str(self.restoration_start.year),
            timestep=timestep,
        )
        yryr = yryr.expand_dims(dim={"metric": [Metric.YrYr]})
        return yryr

    def dNBR(self, timestep: int = 5):
        dnbr = m.dNBR(
            image_stack=self.stack,
            rest_start=str(self.restoration_start.year),
            timestep=timestep,
        )
        dnbr = dnbr.expand_dims(dim={"metric": [Metric.dNBR]})
        return dnbr

    def RRI(self, timestep: int = 5):
        rri = m.RRI(
            image_stack=self.stack,
            rest_start=str(self.restoration_start.year),
            dist_start=str(self.disturbance_start.year),
            timestep=timestep,
        )
        rri = rri.expand_dims(dim={"metric": [Metric.RI]})
        return rri

    def R80P(self, percent_of_target: int = 80):
        recovery_target = self.reference_system.recovery_target()
        r80p = m.R80P(
            image_stack=self.stack,
            rest_start=str(self.restoration_start.year),
            recovery_target=recovery_target["recovery_target"],
            percent=percent_of_target,
        )
        r80p = r80p.expand_dims(dim={"metric": [Metric.R80P]})
        return r80p

    # def metrics(self, metrics: List[str]) -> xr.DataArray:
    #     """Generate recovery metrics over a Restoration Area

    #     Parameters
    #     ----------
    #     metrics : list of str
    #         A list containing the names of metrics to generate.

    #     Returns
    #     -------
    #     metrics_stack : xr.DataArray
    #         A 4D (metric, band, y, x) DataArray, computed metrics values are
    #         located along the `metric` dimension. Coordinates of metrics
    #         are the related enums.Metric.

    #     """
    #     metrics_dict = {}
    #     for metrics_input in metrics:
    #         metric = Metric(metrics_input)
    #         match metric:
    #             case Metric.percent_recovered:
    #                 curr = self.stack.sel(time=self.end_year)
    #                 recovery_target = self.reference_system.recovery_target()
    #                 event = self.stack.sel(time=self.restoration_year)
    #                 metrics_dict[metric] = percent_recovered(
    #                     eval_stack=curr,
    #                     recovery_target=recovery_target["recovery_target"],
    #                     event_obs=event,
    #                 )
    #             case Metric.Y2R:
    #                 filtered_stack = self.stack.sel(
    #                     time=slice(self.restoration_year, self.end_year)
    #                 )
    #                 recovery_target = self.reference_system.recovery_target()
    #                 metrics_dict[metric] = Y2R(
    #                     image_stack=filtered_stack,
    #                     recovery_target=recovery_target["recovery_target"],
    #                     rest_start=str(self.restoration_year.year),
    #                     rest_end=str(self.end_year.year),
    #                 )
    #             case Metric.dNBR:
    #                 metrics_dict[metric] = dNBR(
    #                     restoration_stack=self.stack,
    #                     rest_start=str(self.restoration_year.year),
    #                 )
    #             case Metric.RI:
    #                 metrics_dict[metric] = RI(
    #                     image_stack=self.stack,
    #                     rest_start=str(self.restoration_year.year),
    #                 )
    #     metrics_stack = _stack_bands(
    #         metrics_dict.values(), metrics_dict.keys(), dim_name="metric"
    #     )
    #     return metrics_stack
