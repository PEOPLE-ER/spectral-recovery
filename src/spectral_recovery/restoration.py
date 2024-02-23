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
        # Automatically validate images, polygons, and dates through property setters
        self.full_timeseries = composite_stack
        self.restoration_polygon = restoration_polygon
        self.reference_polygons = reference_polygons
        self.recovery_target_method = recovery_target_method
        self.disturbance_start = self._get_dist_from_frame()
        self.restoration_start = self._get_rest_from_frame()
        self.reference_years = self._get_ref_from_frame()

        # Ensure dates are valid when compared to timeseries and with each other
        self._validate_dates()
        
        self.restoration_image_stack = composite_stack.rio.clip(
            self.restoration_polygon.geometry.values
        )
        self.reference_image_stack = self._get_reference_image_stack()

        self.recovery_target = self.recovery_target_method(
            image_stack=self.reference_image_stack, reference_date=self.reference_years
        )

        self.end_year = pd.to_datetime(self.restoration_image_stack["time"].max().data)
    
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
        RestorationArea.validate_year_orders(self.disturbance_start, self.restoration_start, self.reference_years)

        for years in [self.disturbance_start, self.restoration_start, self.reference_years]:
            # Check that the years are valid for the given timeseries of images
            year_dt = RestorationArea._str_to_dt(years)
            if not self.full_timeseries.satts.contains_temporal(year_dt):
                raise ValueError(
                    f"{year_dt} not contained in the range of the image stack: "
                    f" {self.full_timeseries.time.min().data}-{self.full_timeseries.time.max().data}"
                )

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
