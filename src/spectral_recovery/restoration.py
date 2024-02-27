"""RestorationArea class

The RestorationArea (RA) class is used to coordinate the restoration site
polygon, reference polygons, dates and the timeseries image data. RA 
is intended to be initialized/used by methods returning metrics and/or
restoration info to users (e.g plot_spectral_trajectory or compute_metrics).

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

from spectral_recovery._config import VALID_YEAR


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
        # Automatically validate images, polygons, and dates through property setters
        self.full_timeseries = composite_stack
        self.restoration_polygon = restoration_polygon
        self.reference_polygons = reference_polygons
        self.recovery_target_method = recovery_target_method
        self.disturbance_start = self._get_dist_from_frame()
        self.restoration_start = self._get_rest_from_frame()
        self.reference_years = self._get_ref_from_frame()
        self.timeseries_start = self._npdt_to_year(np.min(self.full_timeseries["time"].data))
        self.timeseries_end = self._npdt_to_year(np.max(self.full_timeseries["time"].data))

        # Ensure dates are valid when compared to timeseries and with each other
        self._validate_dates()
        self._validate_recovery_target_method()
        
        # Clip images with restoration polygon to get rid of urequired data
        self.restoration_image_stack = composite_stack.rio.clip(
            self.restoration_polygon.geometry.values
        )
        # Get the timeseries that will be used for computing recovery targets
        self.reference_image_stack = self._get_reference_image_stack()
        # Compute recovery target based on the passed/default method
        self.recovery_target = self.recovery_target_method(
            image_stack=self.reference_image_stack, reference_date=self.reference_years
        )
    
    full_timeseries = property(operator.attrgetter('_full_timeseries'))
    restoration_polygon = property(operator.attrgetter('_restoration_polygon'))
    reference_polygons = property(operator.attrgetter('_reference_polygons'))
    recovery_target_method = property(operator.attrgetter('_recovery_target_method'))
    disturbance_start = property(operator.attrgetter('_disturbance_start'))
    restoration_start = property(operator.attrgetter('_restoration_start'))
    reference_years = property(operator.attrgetter('_reference_years'))

    @full_timeseries.setter
    def full_timeseries(self, t):
        if not t.satts.is_annual_composite:
            raise ValueError(
                "composite_stack is not a valid set of annual composites. Please"
                " ensure there are no missing years and that the DataArray object"
                " contains 'band', 'time', 'y' and 'x' dimensions."
            ) from None
        self._full_timeseries = t


    @restoration_polygon.setter
    def restoration_polygon(self, rp):
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


    @reference_polygons.setter
    def reference_polygons(self, refp):
        if refp is not None:
            if not self.full_timeseries.satts.contains_spatial(refp):
                raise ValueError(
                    "not all reference_polygons within the bounds of images"
                ) from None

        self._reference_polygons = refp


    @recovery_target_method.setter
    def recovery_target_method(self, rtm):
        if signature(rtm) != expected_signature:
            raise ValueError(
                "The provided recovery target method have the expected call signature:"
                f" {expected_signature} (given {signature(rtm)})"
            )
        self._recovery_target_method = rtm


    @disturbance_start.setter
    def disturbance_start(self, ds):
        RestorationArea._valid_year_format(ds)
        self._disturbance_start = ds


    @restoration_start.setter
    def restoration_start(self, rs):
        RestorationArea._valid_year_format(rs)
        self._restoration_start = rs


    @reference_years.setter
    def reference_years(self, refys):
        if not isinstance(refys, (list, tuple)):
            raise ValueError("Reference years must be a list or tuple")
        if len(refys) != 2:
            raise ValueError(f"Iterable of reference years must contain exactly 2 years (given {len(refys)})")
        for year in refys:
            RestorationArea._valid_year_format(year)
        self._reference_years = refys
    

    def _get_dist_from_frame(self):
        rest_dates = pd.DataFrame(self.restoration_polygon.drop(columns='geometry'))
        try:
            disturbance_start = rest_dates.iloc[:, 0][0]
        except IndexError:
            raise ValueError("Missing disturbance start year. Please ensure the 1st column of the restoration polygon's "
                             " attribute table contains the disturbance window start year. ")
        return str(disturbance_start)
    
    def _get_rest_from_frame(self):
        rest_dates = pd.DataFrame(self.restoration_polygon.drop(columns='geometry'))
        try:
            disturbance_start = rest_dates.iloc[:, 1][0]
        except IndexError:
            raise ValueError("Missing restoration start year. Please ensure the 2nd column of the restoration polygon's "
                             " attribute table contains the restoration window start year. ")
        return str(disturbance_start)
    
    def _get_ref_from_frame(self):
        if self.reference_polygons is not None:
            ref_dates = pd.DataFrame(self.reference_polygons.drop(columns='geometry'))
            try:
                disturbance_start = ref_dates.iloc[:, 0][0]
                disturbance_end = ref_dates.iloc[:, 1][0]
            except:
                ValueError("Missing reference years. Reference start and end years must be"
                        " provided in the 1st and 2nd columns of the reference polygon's"
                        " attribute table.")
        else:
            rest_dates = pd.DataFrame(self.restoration_polygon.drop(columns='geometry'))
            try:
                disturbance_start = rest_dates.iloc[:, 2][0]
                disturbance_end = rest_dates.iloc[:, 3][0]
            except IndexError:
                raise ValueError("Missing reference years. If reference_polygons is None then "
                            " reference years (start and end) must be provided in 3rd and 4th "
                            "columns of the restoration polygon's attribute table." )
        return [str(disturbance_start), str(disturbance_end)]
    
    
    def _validate_dates(self):
        """ Validate reference, disturbance, and restoration dates """
        RestorationArea.validate_year_orders(self.disturbance_start, self.restoration_start, self.reference_years)

        for years in [self.disturbance_start, self.restoration_start, self.reference_years]:
            # Check that the years are valid for the given timeseries of images
            year_dt = RestorationArea._str_to_dt(years)
            if not self.full_timeseries.satts.contains_temporal(year_dt):
                raise ValueError(
                    f"{year_dt} not contained in the range of the image stack: "
                    f" {self.full_timeseries.time.min().data}-{self.full_timeseries.time.max().data}"
                )
    
    def _validate_recovery_target_method(self):
        """ Validate the recovery target method.
        
        If reference polygons are present, ensure that the recovery
        target method is per-polygon, not per-pixel.

        Raises
        ------
        TypeError
            - If reference polygons exist and recovery_target_method
            is an instance of MedianTarget with scale == "pixel"

        """
        if self.reference_polygons is not None:
            if isinstance(self.recovery_target_method, MedianTarget):
                if self.recovery_target_method.scale == "pixel":
                    raise TypeError("Pixel scale median recovery target cannot be used with reference polygons, only polygon scale.")


    def _get_reference_image_stack(self):
        """Get reference image stack.

        Reference image stack is equal to the restoration image
        if not reference polygons are provided, or is clipped from the 
        initial full_timeseries if reference polygons are provided.

        Parameters
        ----------
        reference_polygons : gpd.GeoDataframe
            Reference polygons.
        image_stack : xr.DataArray
            4D stack of images (band, time, y, x).

        Returns
        -------
        reference_stack : xr.DataArray
            5D stack of clipped image data (poly_id, band, time, y, x).
            Coordinate values for poly_id are the row number that each
            polygon belonged to in reference_polygons.

        """
        if self.reference_polygons is None:
            # reference stack is the same as the restoration image stack
            reference_stack = self.restoration_image_stack
        else:
            # reference image stack is clipped using the reference polygons
            # Need to make sure that each polygon is clipped seperately then stacked
            clipped_stacks = {}
            for i, row in self.reference_polygons.iterrows():
                polygon_stack = self.full_timeseries.rio.clip(gpd.GeoSeries(row.geometry).values)
                clipped_stacks[i] = polygon_stack

            reference_stack = xr.concat(
                clipped_stacks.values(),
                dim=pdIndex(clipped_stacks.keys(), name="poly_id"),
            )
        return reference_stack

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


    @staticmethod
    def _npdt_to_year(np_dt: np.datetime64):
        """ Convert np.datetime64 to year str """
        pd_dt = pd.to_datetime(np_dt)
        return str(pd_dt.year)
