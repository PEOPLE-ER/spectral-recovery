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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import Index
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from spectral_recovery.targets import MedianTarget, WindowedTarget, expected_signature
from spectral_recovery.timeseries import _SatelliteTimeSeries
from spectral_recovery.enums import Metric
from spectral_recovery._config import VALID_YEAR

from spectral_recovery import metrics as m


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
        
        if not composite_stack.satts.is_annual_composite:
            raise ValueError(
                "composite_stack is not a valid stack of annual composites. Please"
                " ensure there are no missing years and that the DataArray object"
                " contains 'band', 'time', 'y' and 'x' dimensions."
            ) from None

        self.image_stack = composite_stack
        self.restoration_polygon = _validate_restoration_polygons(
            restoration_polygon=restoration_polygon,
            image_stack=self.image_stack
        )
        (
            self.reference_years,
            self.disturbance_start,
            self.restoration_start,
        ) = _validate_dates(
            reference_years=reference_years,
            disturbance_start=disturbance_start,
            restoration_start=restoration_start,
            image_stack=self.image_stack
        )
        self.restoration_image_stack = self.image_stack.rio.clip(self.restoration_polygon.geometry.values)

        self.reference_polygons = _validate_reference_polygons(reference_polygons=reference_polygons, image_stack=self.image_stack)
        self.recovery_target_method = self._validate_recovery_target_method(recovery_target_method)

        self.reference_image_stack = self._create_reference_image_stack()
        self.recovery_target = self.recovery_target_method(
                self.reference_image_stack, reference_date=self.reference_years
            )
        self.end_year = pd.to_datetime(self.restoration_image_stack["time"].max().data)


    def _validate_recovery_target_method(self, recovery_target_method):
        
        if signature(recovery_target_method) != expected_signature:
            raise ValueError(
                "The provided recovery target method does not have the expected call signature:"
                f" {expected_signature} (given {signature(recovery_target_method)})"
            )
        
        if self.reference_polygons is not None:
            if isinstance(recovery_target_method, MedianTarget):
                if recovery_target_method.scale == "pixel":
                    raise TypeError(
                        "cannot use MedianTarget with scale='pixel' when using"
                        " reference polygons."
                    )
            elif isinstance(recovery_target_method, WindowedTarget):
                raise TypeError(
                    "cannot use WindowedTarget when using reference polygons."
                )
    
        return recovery_target_method    


    def _create_reference_image_stack(self):
        """Create image stack for recovery_target_method.


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
            if isinstance(self.recovery_target_method, MedianTarget):
                reference_stack = self.restoration_image_stack
            elif isinstance(self.recovery_target_method, WindowedTarget):
                orig_time_coords = self.image_stack.time.values
                buffer_size = self.recovery_target_method.N
                buffered_y_coords = self._buffer_coords(self.image_stack.y.values, self.restoration_image_stack.y.values, buffer_size)
                buffered_x_coords = self._buffer_coords(self.image_stack.x.values, self.restoration_image_stack.x.values, buffer_size)
                reference_stack = self.image_stack.loc[dict(time=orig_time_coords, y=buffered_y_coords, x=buffered_x_coords)]
        else: 
            clipped_stacks = {}
            for i, row in self.reference_polygons.iterrows():
                polygon_stack = self.image_stack.rio.clip(gpd.GeoSeries(row.geometry).values)
                clipped_stacks[i] = polygon_stack

            reference_stack = xr.concat(
                clipped_stacks.values(),
                dim=Index(clipped_stacks.keys(), name="poly_id"),
            )
        return reference_stack


    def _compute_recovery_target(self):
        """ Compute recovery target values.
        
        Computes recovery target values using the recovery_target_method. 
        If recovery_target_method is a WindowedTarget method, then the output
        is clipped to the restoration polygon. Otherwise, output is simply
        the return value from the recovery target method.

        This method is needed only as long as WindowedTarget takes and 
        returns unclipped data (i.e no NaNs outside polygon bounds). If
        the WindowedTarget ever handles NaN data, this method can
        be replaced by a call to self.recovery_target_method in __init__.

        Returns
        -------
        recovery_target : xr.DataArray
            Array of recovery target values. Per-band values, with either
            values for every pixel (x,y) that intersects with the restoration
            polygon or a single value for the entire polygon.

        """
        if isinstance(self.recovery_target_method, WindowedTarget):
            recovery_target_no_border = self.recovery_target_method(self.reference_image_stack, reference_date=self.reference_years)
            recovery_target = recovery_target_no_border.rio.clip(self.restoration_polygon.geometry.values)
        else:
            recovery_target = self.recovery_target_method(self.reference_image_stack, reference_date=self.reference_years)
        return recovery_target



    def _buffer_coords(self, full_array_coords, subset_coords, buffer):
        subset_index = np.where(np.in1d(full_array_coords, subset_coords))

        if len(subset_index[0]) == 0:
            raise ValueError("Subset not found in the full array.")
        
        subset_index = subset_index[0][0]  # Extract the index from the tuple
        start_index = max(subset_index - buffer, 0)
        end_index = min(subset_index + len(subset_coords) + buffer, len(full_array_coords))

        buffered_coords = full_array_coords[start_index:end_index]
        return buffered_coords
    
    def y2r(self, percent_of_target: int = 80):
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
        y2r = y2r.expand_dims(dim={"metric": [Metric.Y2R]})
        return y2r


    def yryr(self, timestep: int = 5):
        """Compute the Relative Years to Recovery (YRYR) metric."""
        yryr = m.yryr(
            image_stack=self.restoration_image_stack,
            rest_start=self.restoration_start,
            timestep=timestep,
        )
        yryr = yryr.expand_dims(dim={"metric": [Metric.YRYR]})
        return yryr


    def dnbr(self, timestep: int = 5):
        """Compute the differenced normalized burn ratio (dNBR) metric."""
        dnbr = m.dnbr(
            image_stack=self.restoration_image_stack,
            rest_start=self.restoration_start,
            timestep=timestep,
        )
        dnbr = dnbr.expand_dims(dim={"metric": [Metric.DNBR]})
        return dnbr


    def _rri(self, timestep: int = 5):
        """Compute the relative recovery index (RRI) metric."""
        rri = m.rri(
            image_stack=self.restoration_image_stack,
            rest_start=self.restoration_start,
            dist_start=self.disturbance_start,
            timestep=timestep,
        )
        rri = rri.expand_dims(dim={"metric": [Metric.RRI]})
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
        reference_years = to_dt(self.reference_years)
        restoration_start = to_dt(self.restoration_start)
        disturbance_start = to_dt(self.disturbance_start)

        stats = self.restoration_image_stack.satts.stats()
        stats = stats.sel(
            stats=[
                "median",
                "mean",
            ]
        )
        stats = stats.to_dataframe("value").reset_index()
        stats["time"] = stats["time"].dt.year

        reco_targets = self.recovery_target
        reco_targets = reco_targets.to_dataframe("reco_targets").reset_index()[
            ["band", "reco_targets"]
        ]
        stats = stats.merge(reco_targets, how="left", on="band")
        stats = stats.rename(columns={"stats": "Statistic"})

        # Set theme and colour palette for plots
        sns.set_theme()
        palette = sns.color_palette("deep")

        # Plot per-band statistic lineplots
        with sns.color_palette(palette):
            g = sns.FacetGrid(
                stats,
                col="band",
                hue="Statistic",
                sharey=False,
                sharex=False,
                height=5,
                aspect=1.5,
                legend_out=True,
            )
            g.map_dataframe(sns.lineplot, "time", "value")

        g.set(xticks=stats["time"].unique())
        g.set_xticklabels(rotation=45)

        # Add recovery target line
        g.map_dataframe(
            sns.lineplot,
            "time",
            "reco_targets",
            color="black",
            linestyle="dotted",
            lw=1,
        )
        for ax in g.axes.flat:
            ax.set_xlabel("Year")
        g.axes[0, 0].set_ylabel("Band/Index Value")

        # Plot spectral trajectory windows: reference, disturbance, recovery
        g.map(
            plt.axvline,
            x=restoration_start.year,
            color=palette[2],
            linestyle="dashed",
            lw=1,
        )
        g.map(
            plt.axvline,
            x=disturbance_start.year,
            color=palette[3],
            linestyle="dashed",
            lw=1,
        )
        if hist_ref_sys:
            g.map(
                plt.axvline,
                x=reference_years[0].year,
                color=palette[4],
                linestyle="dashed",
                lw=1,
            )
            if reference_years[1] != disturbance_start:
                g.map(
                    plt.axvline,
                    x=reference_years[1].year,
                    color=palette[4],
                    linestyle="dashed",
                    lw=1,
                )

        for ax in g.axes.flat:
            if hist_ref_sys:
                ax.axvspan(
                    reference_years[0].year,
                    reference_years[1].year,
                    alpha=0.1,
                    color=palette[4],
                )
            ax.axvspan(
                disturbance_start.year,
                restoration_start.year,
                alpha=0.1,
                color=palette[3],
            )
            ax.axvspan(
                restoration_start.year,
                self.end_year.year,
                alpha=0.1,
                color=palette[2],
            )

        # Create custom legend for Facet grid.
        median_line = Line2D([0], [0], color=palette[0], lw=2)
        mean_line = Line2D([0], [0], color=palette[1], lw=2)
        recovery_target_line = Line2D([0], [0], color="black", linestyle="dotted", lw=1)
        recovery_target_patch = Patch(facecolor="black", alpha=0.4)

        recovery_window_line = Line2D(
            [0], [0], color=palette[2], linestyle="dashed", lw=1
        )
        recovery_window_patch = Patch(facecolor=palette[2], alpha=0.1)
        disturbance_window_line = Line2D(
            [0], [0], color=palette[3], linestyle="dashed", lw=1
        )
        disturbance_window_patch = Patch(facecolor=palette[3], alpha=0.1)
        reference_years = Line2D([0], [0], color=palette[4], linestyle="dashed", lw=1)
        reference_years_patch = Patch(facecolor=palette[4], alpha=0.1)

        custom_handles = [
            median_line,
            mean_line,
            (disturbance_window_line, disturbance_window_patch),
            (recovery_window_line, recovery_window_patch),
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
                        (recovery_target_line, recovery_target_patch),
                    )
                else:
                    custom_handles.insert(
                        2,
                        recovery_target_line,
                    )
            custom_handles.insert(3, (reference_years, reference_years_patch))
            labels.insert(2, "historic recovery target (median)")
            labels.insert(3, "reference year(s)")
        else:
            custom_handles.insert(
                2,
                recovery_target_line,
            )
            labels.insert(2, "reference recovery target")

        plt.figlegend(
            labels=labels,
            handles=custom_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            ncol=6,
        )
        plt.suptitle("Spectral Trajectory of RestorationArea Site")
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
