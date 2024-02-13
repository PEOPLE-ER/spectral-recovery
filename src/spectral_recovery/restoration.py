"""Restoration Area and Reference System classes.

The RestorationArea class represents a restoration event and contains
methods for computing spectral recovery metrics. Users create a
RestorationArea by providing a restoration polygon, reference polygon,
event dates, and a stack of annual composites. 

A RestorationArea contains a ReferenceSystem, which is a class that
represents the reference area(s) and contains methods for computing
the recovery target. 

"""

from typing import Callable, Optional, Union, List, Tuple
from datetime import datetime
from inspect import signature

import xarray as xr
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import Index
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerPatch

from spectral_recovery.targets import MedianTarget, expected_signature
from spectral_recovery.timeseries import _SatelliteTimeSeries
from spectral_recovery.enums import Metric, Index
from spectral_recovery._config import VALID_YEAR

from spectral_recovery import metrics as m


def _get_reference_image_stack(reference_polygons, image_stack):
    """Clip reference polygon data, stack along new poly_id dim.

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
    clipped_stacks = {}
    for i, row in reference_polygons.iterrows():
        polygon_stack = image_stack.rio.clip(gpd.GeoSeries(row.geometry).values)
        clipped_stacks[i] = polygon_stack

    reference_stack = xr.concat(
        clipped_stacks.values(),
        dim=Index(clipped_stacks.keys(), name="poly_id"),
    )
    return reference_stack


def validate_year_format(year_str, field_name):
    year_match = VALID_YEAR.match(year_str)
    if not year_match:
        raise ValueError(
            f"Could not parse {field_name} ({year_str}) into a year. "
            "Please ensure the year is in the format 'YYYY'."
        )


def parse_and_validate_date(date_str, field_name):
    if not isinstance(date_str, str):
        raise TypeError(f"{field_name} must be a string.")

    validate_year_format(date_str, field_name)
    return date_str


def validate_year_order(disturbance_start, restoration_start):
    if restoration_start < disturbance_start:
        raise ValueError(
            "The disturbance start year must be less than the restoration start year."
        )


def process_date_fields(disturbance_start, restoration_start):
    if disturbance_start is not None:
        disturbance_start = parse_and_validate_date(
            disturbance_start, "disturbance_start"
        )
        if restoration_start is None:
            restoration_start = str(int(disturbance_start) + 1)
            validate_year_order(disturbance_start, restoration_start)

    if restoration_start is not None:
        restoration_start = parse_and_validate_date(
            restoration_start, "restoration_start"
        )
        if disturbance_start is None:
            disturbance_start = str(int(restoration_start) - 1)

    validate_year_order(disturbance_start, restoration_start)

    return disturbance_start, restoration_start


def process_reference_years(reference_years):
    if isinstance(reference_years, str):
        reference_years = parse_and_validate_date(reference_years, "reference_years")
    else:
        try:
            _ = iter(reference_years)
            if len(reference_years) == 2:
                reference_years = [
                    parse_and_validate_date(date_str, "reference_years")
                    for date_str in reference_years
                ]
            else:
                raise ValueError(
                    "reference_years must be a string or iterable of 2 strings."
                )
        except TypeError:
            raise TypeError(
                "reference_years must be a string or iterable of 2 strings."
            )

    return reference_years


def check_years_against_images(year, image_stack):
    if not image_stack.satts.contains_temporal(year):
        raise ValueError(
            f"{year} contained in the range of the image stack,"
            f" {image_stack.time.min().data}-{image_stack.time.max().data}"
        )


def to_dt(years: str | List[str]):
    if isinstance(years, list):
        years_dt = [0, 0]
        for i, year in enumerate(years):
            years_dt[i] = pd.to_datetime(year)
    else:
        years_dt = pd.to_datetime(years)
    return years_dt


def _validate_dates(reference_years, disturbance_start, restoration_start, image_stack):
    if disturbance_start is None and restoration_start is None:
        raise ValueError(
            "At least one of disturbance_start or restoration_start needs to be set"
            " (both are None)"
        )

    disturbance_start, restoration_start = process_date_fields(
        disturbance_start, restoration_start
    )
    reference_years = process_reference_years(reference_years)

    for years in [disturbance_start, restoration_start, reference_years]:
        check_years_against_images(to_dt(years), image_stack)

    return reference_years, disturbance_start, restoration_start


def _validate_restoration_polygons(restoration_polygon, image_stack):
    if restoration_polygon.shape[0] != 1:
        raise ValueError(
            "restoration_polygons contains more than one Polygon."
            "A RestorationArea instance can only contain one Polygon."
        ) from None
    if not image_stack.satts.contains_spatial(restoration_polygon):
        raise ValueError("restoration_polygon is not within the bounds of images") from None

    return restoration_polygon

def _validate_reference_polygons(reference_polygons, image_stack):
    if reference_polygons is not None:
        if not image_stack.satts.contains_spatial(reference_polygons):
            raise ValueError("not all reference_polygons within the bounds of images") from None
        
    return reference_polygons

class RestorationArea:
    """A Restoration Area (RA).

    Attributes
    -----------
    restoration_polygon : GeoDataFrame
        The spatial deliniation of the restoration event. There
        must only be one geometry in the GeoDataframe and it must be
        of type shapely.Polygon or shapely.MultiPolygon.
    composite_stack : xr.DataArray
        A 4D (band, time, y, x) DataArray of images.
    disturbance_start : str or datetime
        The start year of the disturbance window. If None, defaults
        to the year prior to restoration_start.
    restoration_start : str or datetime
        The start year of the recovery window. If None, defaults
        to the year following disturbance_start.
    reference_polygon : GeoDataFrame
        The spatial delinitation of the reference area(s).
    reference_years : datetime or Tuple of datetimes
        The year or range of years of the reference window.
    recovery_target_method : callable
        The method used for computing the recovery target. Default
        is median target method with polygon scale.
    recovery_target : xr.DataArray
        The recovery target values.

    """

    def __init__(
        self,
        restoration_polygon: gpd.GeoDataFrame,
        reference_years: str | List[str],
        composite_stack: xr.DataArray,
        reference_polygons: gpd.GeoDataFrame = None,
        disturbance_start: str = None,
        restoration_start: str = None,
        recovery_target_method: Callable[
            [xr.DataArray, Tuple[datetime]], xr.DataArray
        ] = MedianTarget(scale="polygon"),
    ) -> None:
        
        if composite_stack.satts.is_annual_composite:
            self.restoration_polygon = _validate_restoration_polygons(
                restoration_polygon=restoration_polygon, image_stack=composite_stack
            )
            (
                self.reference_years,
                self.disturbance_start,
                self.restoration_start,
            ) = _validate_dates(
                reference_years=reference_years,
                disturbance_start=disturbance_start,
                restoration_start=restoration_start,
                image_stack=composite_stack,
            )
            self.stack = composite_stack.rio.clip(
                self.restoration_polygon.geometry.values
            )
        else:
            raise ValueError(
                "composite_stack is not a valid stack of annual composites. Please"
                " ensure there are no missing years and that the DataArray object"
                " contains 'band', 'time', 'y' and 'x' dimensions."
            ) from None

        if signature(recovery_target_method) != expected_signature:
            raise ValueError(
                "The provided recovery target method have the expected call signature:"
                f" {expected_signature} (given {signature(recovery_target_method)})"
            )
        self.recovery_target_method = recovery_target_method

        self.reference_polygons = _validate_reference_polygons(reference_polygons=reference_polygons, image_stack=composite_stack)
        if self.reference_polygons is None:
            reference_image_stack = self.stack
        else:  # computing recovery target using reference polygons
            if isinstance(recovery_target_method, MedianTarget):
                if recovery_target_method.scale == "pixel":
                    raise TypeError(
                        "cannot use MedianTarget with scale='pixel' when using"
                        " reference polygons."
                    )
            reference_image_stack = _get_reference_image_stack(
                reference_polygons=self.reference_polygons,
                image_stack=composite_stack,
            )

        self.recovery_target = recovery_target_method(
            reference_image_stack, reference_date=self.reference_years
        )

        self.end_year = pd.to_datetime(self.stack["time"].max().data)

    def y2r(self, percent_of_target: int = 80):
        """Compute the Years to Recovery (Y2R) metric."""
        post_restoration = self.stack.sel(
            time=slice(self.restoration_start, self.end_year)
        )
        y2r = m.y2r(
            image_stack=post_restoration,
            recovery_target=self.recovery_target,
            rest_start=self.restoration_start,
            percent=percent_of_target,
        )
        y2r = y2r.expand_dims(dim={"metric": [Metric.Y2R]})
        return y2r

    def yryr(self, timestep: int = 5):
        """Compute the Relative Years to Recovery (YRYR) metric."""
        yryr = m.yryr(
            image_stack=self.stack,
            rest_start=self.restoration_start,
            timestep=timestep,
        )
        yryr = yryr.expand_dims(dim={"metric": [Metric.YRYR]})
        return yryr

    def dnbr(self, timestep: int = 5):
        """Compute the differenced normalized burn ratio (dNBR) metric."""
        dnbr = m.dnbr(
            image_stack=self.stack,
            rest_start=self.restoration_start,
            timestep=timestep,
        )
        dnbr = dnbr.expand_dims(dim={"metric": [Metric.DNBR]})
        return dnbr

    def _rri(self, timestep: int = 5):
        """Compute the relative recovery index (RRI) metric."""
        rri = m.rri(
            image_stack=self.stack,
            rest_start=self.restoration_start,
            dist_start=self.disturbance_start,
            timestep=timestep,
        )
        rri = rri.expand_dims(dim={"metric": [Metric.RRI]})
        return rri

    def r80p(self, percent_of_target: int = 80, timestep: int = 5):
        """Compute the recovery to 80% of target (R80P) metric."""
        r80p = m.r80p(
            image_stack=self.stack,
            rest_start=self.restoration_start,
            recovery_target=self.recovery_target,
            timestep=timestep,
            percent=percent_of_target,
        )
        r80p = r80p.expand_dims(dim={"metric": [Metric.R80P]})
        return r80p


    def plot_spectral_trajectory(self, path: str = None) -> None:
        """Create spectral trajectory plot of the RestorationArea

        Parameters
        ----------
        path : str, optional
            The path to save the plot to.
        """
        hist_ref_sys = self.reference_polygons == None

        stats = self.stack.satts.stats()
        stats = stats.sel(
            stats=[
                "median",
                "mean",
            ]
        )

        stats = stats.assign_coords(band=([str(b) for b in stats.band.values]))
        stats = stats.to_dataframe("value")
        
        recovery_target = self.recovery_target.assign_coords(band=([str(b) for b in self.recovery_target.band.values]))
        reco_targets = recovery_target.to_dataframe("reco_targets").dropna(how="any")

        data = stats.merge(reco_targets, left_index=True, right_index=True)[["value", "reco_targets"]]
        data = data.reset_index()
        data["time"] = data["time"].apply(lambda x: str(x.year))

        # Set theme and colour palette for plots
        sns.set_theme()
        palette = sns.color_palette("deep")

        bands = data["band"].unique()
        fig, axs = plt.subplots(len(bands), 1, sharey=False, sharex=True, figsize=[10, 7.5])
        # fig.supylabel("Band/Index Value")
        # Plot per-band statistic lineplots
        for i, band in enumerate(bands):
            band_data = data[data["band"] == band]
            try: 
                axi = axs[i]
            except TypeError:
                axi = axs

            sns.lineplot(data=band_data, x="time", hue="stats", y="value", ax=axi, legend=False, lw=1)
            sns.lineplot(data=band_data[band_data["stats"]=="mean"], x="time", y="reco_targets", ax=axi, color="black", linestyle=(0, (3, 5, 1, 5)), lw=1)

            axi.set_title(band)
            axi.set_xticks(axi.get_xticks(), data["time"].unique().tolist(), rotation=45, ha='right')
            axi.set_xlabel("Year")
            axi.set_ylabel("Band/Index Value")

            axi.axvline(
                x=self.restoration_start,
                color=palette[2],
                linestyle="dashed",
                lw=1,
            )
            axi.axvspan(
                self.restoration_start,
                str(self.end_year.year),
                alpha=0.2,
                color=palette[2],
            )

            axi.axvline(
                x=self.disturbance_start,
                color=palette[3],
                linestyle="dashed",
                lw=1,
            )
            axi.axvspan(
                self.disturbance_start,
                self.restoration_start,
                alpha=0.2,
                color=palette[3],
            )

            if hist_ref_sys:
                axi.axvline(
                x=self.reference_years[0],
                color=palette[4],
                linestyle="dashed",
                lw=1,

                )
                axi.axvspan(
                    self.reference_years[0],
                    self.reference_years[1],
                    alpha=0.2,
                    color=palette[4],
                )
                if self.reference_years[1] != self.disturbance_start:
                    axi.axvline(
                    x=self.reference_years[1],
                    color=palette[4],
                    linestyle="dashed",
                    lw=1,
                    ) 

        labels, custom_handles, = _create_custom_legend(self, palette, hist_ref_sys)

        if path is None:
            plt.figlegend(
                labels=labels,
                handles=custom_handles,
                loc="lower center",
                fancybox=True,
                ncol=3,
                handler_map={Patch: HandlerFilledBetween()}
            )
            plt.tight_layout()
            plt.subplots_adjust(bottom=plt.rcParams["figure.subplot.bottom"]+0.07)
        else:
            plt.figlegend(
                labels=labels,
                handles=custom_handles,
                bbox_to_anchor=(0.5, -0.075),
                loc="lower center",
                fancybox=True,
                ncol=3,
                handler_map={Patch: HandlerFilledBetween()}
            )
            plt.tight_layout()

        if path:
            plt.savefig(path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

# def _

def _create_custom_legend(self, palette, hist_ref_sys) -> Tuple[List, List]:
    """ Create a custom legend to match trajectory plots 
    
    Returns
    -------
    tuple of lists 
        custom labels and handles to pass to ``figlegend``
    
    """
    median_line = Line2D([0], [0], color=palette[0], lw=2)
    mean_line = Line2D([0], [0], color=palette[1], lw=2)
    recovery_target_line = Line2D([0], [0], color="black", linestyle=(0, (3, 5, 1, 5)), lw=1)
    recovery_window_patch = Patch(facecolor=palette[2], alpha=0.2)
    disturbance_window_patch = Patch(facecolor=palette[3], alpha=0.2)
    reference_years_patch = Patch(facecolor=palette[4], alpha=0.2)

    custom_handles = [
        median_line,
        mean_line,
        disturbance_window_patch,
        recovery_window_patch,
    ]

    labels = [
        "median",
        "mean",
        "disturbance window",
        "recovery window",
    ]
    if hist_ref_sys:
        if isinstance(self.recovery_target_method, MedianTarget):
            if self.recovery_target_method.scale == "pixel":
                custom_handles.insert(
                    2,
                    (recovery_target_line),
                )
                labels.insert(2, "recovery target (estimated mean)")
            else:
                custom_handles.insert(
                    2, 
                    recovery_target_line,
                )
                labels.insert(2, "recovery target")

        custom_handles.insert(3, reference_years_patch)
        labels.insert(3, "reference year(s)")
    else:
        custom_handles.insert(
            2,
            recovery_target_line,
        )
        labels.insert(2, "recovery target")

    return labels, custom_handles

class HandlerFilledBetween(HandlerPatch):
    """ Custom Patch Handler for trajectory windows.

    Draws Patch objects with left and right edges coloured/dashed
    to match the style of trajectory window Patches in the plots.
    
    """
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        p = super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)[0]
        color = p.get_facecolor()
        x0, y0 = 0, 0
        x1 = x0 + width
        y1 = y0 + height
        line_left = Line2D([x0, x0], [y0, y1], color=color, linestyle="dashed", lw=0.85, alpha=1)
        line_right = Line2D([x1, x1], [y0, y1], color=color, linestyle="dashed", lw=0.85, alpha=1)
        return [p, line_left, line_right]