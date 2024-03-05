"""Restoration Area and Reference System classes.

The RestorationArea class represents a restoration event and contains
methods for computing spectral recovery metrics. Users create a
RestorationArea by providing a restoration polygon, reference polygon,
event dates, and a stack of annual composites. 

A RestorationArea contains a ReferenceSystem, which is a class that
represents the reference area(s) and contains methods for computing
the recovery target. 

"""
import operator 

from typing import Callable, Dict, List, Tuple
from datetime import datetime
from inspect import signature

import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np

from pandas import Index as pdIndex

from spectral_recovery.targets import MedianTarget, expected_signature
from spectral_recovery.timeseries import _SatelliteTimeSeries
from spectral_recovery.indices import compute_indices
from spectral_recovery.enums import Metric
from spectral_recovery._config import VALID_YEAR, SUPPORTED_PLATFORMS

from spectral_recovery import metrics as m


def compute_metrics(
        timeseries_data: xr.DataArray,
        restoration_polygons: gpd.GeoDataFrame,
        metrics: List[str],
        indices: List[str],
        reference_polygons: gpd.GeoDataFrame = None,
        index_constants: Dict[str, int] = {},
        timestep: int = 5, 
        percent_of_target: int = 80,
        recovery_target_method = MedianTarget(scale="polygon"),
    ):

    indices_stack = compute_indices(image_stack=timeseries_data, indices=indices, constants=index_constants)
    restoration_area = RestorationArea(
        restoration_polygon=restoration_polygons,
        reference_polygons=reference_polygons,
        composite_stack=indices_stack,
        recovery_target_method=recovery_target_method
    )
    m_results = []
    for m in metrics:
        m_func = getattr(restoration_area, m.lower())
        m_results.append(m_func(timestep=timestep, percent_of_target=percent_of_target))

    metrics = xr.concat(m_results, "metric")

    return metrics


class RestorationArea:
    """A Restoration Area (RA).

    A RestorationArea object validates and holds the locations, spectral
    data, dates, and recovery target(s) that define a restoration area.

    Attributes
    -----------
    restoration_polygon : GeoDataFrame
        The spatial deliniation of the restoration event. There
        must only be one geometry in the GeoDataframe and it must be
        of type shapely.Polygon or shapely.MultiPolygon.
    composite_stack : xr.DataArray
        A 4D (band, time, y, x) DataArray of images.
    reference_polygon : GeoDataFrame
        The spatial delinitation of the reference area(s).
    recovery_target_method : callable
        The method used for computing the recovery target. Default
        is median target method with polygon scale.
    recovery_target : xr.DataArray
        The recovery target values.

    """

    def __init__(
        self,
        restoration_polygon: gpd.GeoDataFrame,
        composite_stack: xr.DataArray,
        reference_polygons: gpd.GeoDataFrame = None,
        recovery_target_method: Callable[
            [xr.DataArray, Tuple[datetime]], xr.DataArray
        ] = MedianTarget(scale="polygon"),
    ) -> None:
        self.full_timeseries = composite_stack
        self.reference_polygons = reference_polygons
        self.restoration_polygon = restoration_polygon
        self.recovery_target_method = recovery_target_method
    
    @property
    def full_timeseries(self):
        return self._full_timeseries
    
    @full_timeseries.setter
    def full_timeseries(self, t):
        """ full_timeseries setter. 
        
        Checks that the provided timeseries contains the correct
        dims and is an annual timeseries before setting the property.
        Forces lazy (re-)computation of the dependant cached properties,
        restoration_image_stack and reference_image_stack.
        
        """
        self._restoration_image_stack = None
        self._reference_image_stack = None

        if not t.satts.is_annual_composite:
            raise ValueError(
                "composite_stack is not a valid set of annual composites. Please"
                " ensure there are no missing years and that the DataArray object"
                " contains 'band', 'time', 'y' and 'x' dimensions."
            ) from None
        self._full_timeseries = t

    @property
    def end_year(self):
        """The final year of the timeseries"""
        return pd.to_datetime(self.full_timeseries["time"].max().data)
    
    @property
    def start_year(self):
        """The first year of the timeseries"""
        return pd.to_datetime(self.full_timeseries["time"].min().data)
    
    @property
    def restoration_polygon(self):
        """Restoration site GeoDataFrame
        
        GeoDataFrame contains a Shapely.Polygon and
        2 or 4 str columns. Polygon defines the area of
        the restoration site while attributes define 
        the disturbance start, restoration start, reference
        end, and reference start years respectively.
        
        """
        return self._restoration_polygon
    
    @restoration_polygon.setter
    def restoration_polygon(self, rp):
        """ restoration_polygon setter.

        Checks that input is a GeoDataFrame, contains
        only one row/geometry and that the given geom is
        within the bounds of self.full_timeseries. Immediately
        (re-)computates disturbance_start, restoration_start, 
        and the reference year cached properties. Forces lazy
        recomputation of recovery_target.

        """
        self._recovery_target = None
        self._disturbance_start = None
        self._restoration_start = None
        if self._reference_polygons is None:
            self._reference_years = None
        
        if not isinstance(rp, gpd.GeoDataFrame):
            raise ValueError(f"restoration_polygon must be a GeoDataFrame (recieved type {type(rp)})")
        if rp.shape[0] != 1:
            raise ValueError(
                "restoration_polygon instance can only contain one Polygon."
            ) from None
        if not self.full_timeseries.satts.contains_spatial(rp):
            raise ValueError(
                "restoration_polygon is not within the bounds of images"
            ) from None

        self._restoration_polygon = rp

        # Trigger eager computation of date attributes.
        # Make any incompatible dates throw an error early
        # NOTE: this is messy because restoration_polygon now takes
        # the whole DF with the dates (not just poly), which makes
        # the date attributes dependant on it. This attr should
        # be refactored for clarity. 
        self.disturbance_start
        self.restoration_start
        self.reference_years

    
    @property
    def reference_polygons(self):
        """Reference system GeoDataFrame

        GeoDataFrame contains a Shapely.Polygon and
        2 str columns. Polygon defines the areas used in
        the reference system while columns define 
        the reference end and start year, respectively.
        
        """
        return self._reference_polygons
    
    @reference_polygons.setter
    def reference_polygons(self, refp):
        self._reference_years = None
        self._reference_image_stack = None
        if refp is not None:
            if not self.full_timeseries.satts.contains_spatial(refp):
                raise ValueError(
                    "not all reference_polygons within the bounds of images"
                ) from None
        self._reference_polygons = refp
    
    @property
    def recovery_target_method(self):
        """Method used to compute recovery targets for the RestorationArea"""
        return self._recovery_target_method
    
    @recovery_target_method.setter
    def recovery_target_method(self, rtm):
        self._recovery_target = None
        if signature(rtm) != expected_signature:
            raise ValueError(
                "The provided recovery target method must have the expected call signature:"
                f" {expected_signature} (given {signature(rtm)})"
            )
        if self.reference_polygons is not None:
            if isinstance(rtm, MedianTarget):
                if rtm.scale == "pixel":
                    raise TypeError("Pixel scale median recovery target cannot be used with reference polygons, only polygon scale.")
        self._recovery_target_method = rtm
    
    @property
    def recovery_target(self):
        """Recovery target of the RestorationArea.

        The recovery targets of the RestorationArray provided in
        an xarray.DataArray. The targets are computed using the
        recovery_target_method and based on either historic conditions
        or a reference system, if provided.

        """
        if self._recovery_target is None:
            self._recovery_target = self.recovery_target_method(
                image_stack=self.reference_image_stack, reference_date=self.reference_years
            )
        return self._recovery_target
    
    @property
    def disturbance_start(self):
        """Start year of the disturbance window."""
        if self._disturbance_start is None:
            self._disturbance_start = self._get_dist_from_frame()
            if self._disturbance_start >= self.restoration_start:
                raise ValueError(
                    f"Disturbance start year must be strictly less than the restoration start year (given disturbance_start={self._disturbance_start} and restoration_start={self.restoration_start})"
                )
            if not self.full_timeseries.satts.contains_temporal(RestorationArea._str_to_dt(self._disturbance_start)):
                raise ValueError(
                    f"Restoration start year { self._disturbance_start} not within timeseries range of {self.start_year}-{self.end_year}."
                )
        return self._disturbance_start

    @property
    def restoration_start(self):
        """Start year of the restoration window."""
        if self._restoration_start is None:
            self._restoration_start = self._get_rest_from_frame()
            if self._restoration_start <= self.disturbance_start:
                raise ValueError(
                    f"Disturbance start year must be strictly less than the restoration start year (given disturbance_start={self.disturbance_start} and restoration_start={self._restoration_start})"
                )
            if not self.full_timeseries.satts.contains_temporal(RestorationArea._str_to_dt(self._restoration_start)):
                raise ValueError(
                    f"Restoration start year { self._restoration_start} not within timeseries range of {self.start_year}-{self.end_year}."
                )
        return self._restoration_start
    
    @property
    def reference_years(self):
        """Start and end year of the reference window."""
        if self._reference_years is None:
            self._recovery_target = None
            self._reference_years = self._get_ref_from_frame()
            if  self._reference_years[0] > self._reference_years[1]:
                raise ValueError(
                    f"Reference start year must be less than or equal to end year (but {self._reference_years[0]} > {self._reference_years[1]})"
                )
            if not self.full_timeseries.satts.contains_temporal(RestorationArea._str_to_dt(self._reference_years)):
                raise ValueError(
                    f"Reference years { self._reference_years} not within timeseries range of {self.start_year}-{self.end_year}."
                )
        return self._reference_years
    
    @property
    def reference_image_stack(self):
        """Reference image stack.

        The image stack for computing recovery targets. The reference 
        image stack is the same as the restoration image stack if
        using historic targets. If using a reference system, the reference
        image stack is clipped from the given timeseries using the 
        reference polygons. 

        """
        if self._reference_image_stack is None:
            self._recovery_target = None
            if self.reference_polygons is None:
                # reference stack is the same as the restoration image stack
                 self._reference_image_stack = self.restoration_image_stack
            else:
                # reference image stack is clipped using the reference polygons
                # Need to make sure that each polygon is clipped seperately then stacked
                clipped_stacks = {}
                for i, row in self.reference_polygons.iterrows():
                    polygon_stack = self.full_timeseries.rio.clip(gpd.GeoSeries(row.geometry).values)
                    clipped_stacks[i] = polygon_stack

                self._reference_image_stack = xr.concat(
                    clipped_stacks.values(),
                    dim=pdIndex(clipped_stacks.keys(), name="poly_id"),
                )

        return self._reference_image_stack
    
    @property
    def restoration_image_stack(self):
        """Restoration image stack.

        The image stack containing the restoration site. 
        Clipped to the bounding box of the restoration polygon.

        """
        if self._restoration_image_stack is None:
            self.full_timeseries.rio.clip(
                self.restoration_polygon.geometry.values
            )
        return self._restoration_image_stack

    def _get_dist_from_frame(self):
        rest_dates = pd.DataFrame(self.restoration_polygon.drop(columns='geometry'))
        try:
            disturbance_start = rest_dates.iloc[:, 0][0]
        except IndexError:
            raise ValueError("Missing disturbance start year. Please ensure the 1st column of the restoration polygon's "
                             " attribute table contains the disturbance window start year. ")
        disturbance_start_str = str(disturbance_start)
        RestorationArea._valid_year_format(disturbance_start_str)
        return disturbance_start_str
    
    def _get_rest_from_frame(self):
        rest_dates = pd.DataFrame(self.restoration_polygon.drop(columns='geometry'))
        try:
            restoration_start = rest_dates.iloc[:, 1][0]
        except IndexError:
            raise ValueError("Missing restoration start year. Please ensure the 2nd column of the restoration polygon's "
                             " attribute table contains the restoration window start year. ")
        rest_start_str = str(restoration_start)
        RestorationArea._valid_year_format(rest_start_str)
        return rest_start_str
    
    def _get_ref_from_frame(self):
        if self.reference_polygons is not None:
            ref_dates = pd.DataFrame(self.reference_polygons.drop(columns='geometry'))
            try:
                ref_start = ref_dates.iloc[:, 0][0]
                ref_end = ref_dates.iloc[:, 1][0]
            except:
                ValueError("Missing reference years. Reference start and end years must be"
                        " provided in the 1st and 2nd columns of the reference polygon's"
                        " attribute table.")
        else:
            ref_dates = pd.DataFrame(self.restoration_polygon.drop(columns='geometry'))
            try:
                ref_start = ref_dates.iloc[:, 2][0]
                ref_end = ref_dates.iloc[:, 3][0]
            except IndexError:
                raise ValueError("Missing reference years. If reference_polygons is None then "
                            " reference years (start and end) must be provided in 3rd and 4th "
                            "columns of the restoration polygon's attribute table." )
        
        ref_start_str = str(ref_start)
        ref_end_str = str(ref_end)
        RestorationArea._valid_year_format(ref_start_str)
        RestorationArea._valid_year_format(ref_end_str)

        return [ref_start_str, ref_end_str]    

    @staticmethod
    def _valid_year_format(year_str):
        year_match = VALID_YEAR.match(year_str)
        if not year_match:
            raise ValueError(
                f"Could not parse {year_str} into a year. "
                "Please ensure the year is in the format 'YYYY'."
            )

    @staticmethod
    def validate_year_orders(disturbance_start, restoration_start, reference_years):

        if reference_years[0] > reference_years[1]:
                raise ValueError(
                    f"Reference start year must be less than or equal to the reference end year (but {reference_years[0]} > {reference_years[1]})"
                )

        if restoration_start <= disturbance_start:
            raise ValueError(
                "The disturbance start year must be strictly less than the restoration start year."
            )

    @staticmethod
    def _str_to_dt(years: str | List[str]):
        if isinstance(years, list):
            years_dt = [0, 0]
            for i, year in enumerate(years):
                years_dt[i] = pd.to_datetime(year)
        else:
            years_dt = pd.to_datetime(years)
        return years_dt

    def y2r(self, timestep = 5, percent_of_target: int = 80):
        """Compute the Years to Recovery (Y2R) metric."""
        post_restoration = self.restoration_image_stack.sel(
            time=slice(self.restoration_start, self.end_year)
        )
        y2r = m.y2r(
            image_stack=post_restoration,
            recovery_target=self.recovery_target,
            rest_start=self.restoration_start,
            percent=percent_of_target,
        )
        y2r = y2r.expand_dims(dim={"metric": [str(Metric.Y2R)]})
        return y2r

    def yryr(self, timestep: int = 5, percent_of_target: int = 80):
        """Compute the Relative Years to Recovery (YRYR) metric."""
        yryr = m.yryr(
            image_stack=self.restoration_image_stack,
            rest_start=self.restoration_start,
            timestep=timestep,
        )
        yryr = yryr.expand_dims(dim={"metric": [str(Metric.YRYR)]})
        return yryr

    def dnbr(self, timestep: int = 5, percent_of_target: int = 80):
        """Compute the differenced normalized burn ratio (dNBR) metric."""
        dnbr = m.dnbr(
            image_stack=self.restoration_image_stack,
            rest_start=self.restoration_start,
            timestep=timestep,
        )
        dnbr = dnbr.expand_dims(dim={"metric": [str(Metric.DNBR)]})
        return dnbr

    def _rri(self, timestep: int = 5, percent_of_target: int = 80):
        """Compute the relative recovery index (RRI) metric."""
        rri = m.rri(
            image_stack=self.restoration_image_stack,
            rest_start=self.restoration_start,
            dist_start=self.disturbance_start,
            timestep=timestep,
        )
        rri = rri.expand_dims(dim={"metric": [str(Metric.RRI)]})
        return rri

    def r80p(self, percent_of_target: int = 80, timestep: int = 5):
        """Compute the recovery to 80% of target (R80P) metric."""
        r80p = m.r80p(
            image_stack=self.restoration_image_stack,
            rest_start=self.restoration_start,
            recovery_target=self.recovery_target,
            timestep=timestep,
            percent=percent_of_target,
        )
        r80p = r80p.expand_dims(dim={"metric": [str(Metric.R80P)]})
        return r80p
