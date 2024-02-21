"""Restoration Area and Reference System classes.

The RestorationArea class represents a restoration event and contains
methods for computing spectral recovery metrics. Users create a
RestorationArea by providing a restoration polygon, reference polygon,
event dates, and a stack of annual composites. 

A RestorationArea contains a ReferenceSystem, which is a class that
represents the reference area(s) and contains methods for computing
the recovery target. 

"""

from typing import Callable, Dict, List, Tuple
from datetime import datetime
from inspect import signature

import xarray as xr
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import Index as pdIndex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerPatch

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
        dim=pdIndex(clipped_stacks.keys(), name="poly_id"),
    )
    return reference_stack


def validate_year_format(year_str, field_name):
    year_match = VALID_YEAR.match(year_str)
    if not year_match:
        raise ValueError(
            f"Could not parse {field_name} ({year_str}) into a year. "
            "Please ensure the year is in the format 'YYYY'."
        )

def validate_year_order(disturbance_start, restoration_start):

    if restoration_start < disturbance_start:
        raise ValueError(
            "The disturbance start year must be less than the restoration start year."
        )

def process_reference_years(reference_years):
    
    validate_year_format(reference_years[0], "reference start")
    validate_year_format(reference_years[1], "reference end")
    if reference_years[0] is not None:
        if reference_years[0] > reference_years[1]:
            raise ValueError(
                f"The reference start year must be less than or equal to the reference end year (but {reference_years[0]} > {reference_years[1]})"
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


def _get_dates_from_frame(rest_frame: gpd.GeoDataFrame, ref_frame: gpd.GeoDataFrame):
    """Read restoration polygon dates from GeoDataframes.

    Get disturbance window start, restoration window start,
    refernce year start, and reference year end. If just restoration
    frame is given, then the aforementioned years are read from the
    rest_frame's 1st, 2nd, 3rd, and 4th columns respectively. If both
    rest and ref_frames are given, then disturbance and restoration
    start years are taken from the 1st and 2nd columns of rest_frame
    and reference years are taken from 1st and 2nd column of ref_frame.

    Parameters
    ----------
    rest_frame : gpd.GeoDataFrame
        The restoration polygon dataframe
    ref_frame : gpd.GeoDataFrame
        The reference polygon dataframe

    Returns
    -------
    disturbance_start : str
        The start year of the disturbance window.
    restoration_start : str
        The start year of the restoration window.
    reference_years : list of str
        The start and end year of the reference window.

    """
    rest_frame = pd.DataFrame(rest_frame.drop(columns='geometry'))
    if ref_frame is not None:
        ref_frame = pd.DataFrame(ref_frame.drop(columns='geometry'))
    try:
        disturbance_start = rest_frame.iloc[:, 0][0]
        restoration_start = rest_frame.iloc[:, 1][0]
        try:
            reference_year_start = rest_frame.iloc[:, 2][0]
            reference_year_end = rest_frame.iloc[:, 3][0]
        except IndexError:
            if ref_frame is None:
                raise ValueError(
                    "Missing reference years. Reference years (start and end) must be"
                    " provided in 3rd and 4th columns of the restoration polygon's"
                    " attribute table."
                )
            else:
                try:
                    reference_year_start = ref_frame.iloc[:, 0][0]
                    reference_year_end = ref_frame.iloc[:, 1][0]
                except IndexError:
                    raise ValueError(
                        "Missing reference years. Reference years (start and end) must be"
                        " provided in the 1st and 2nd columns of the reference polygon's"
                        " attribute table."
                    )

    except IndexError:
        raise ValueError(
            "Missing disturbance or restoration start years. Please ensure the 1st and"
            " 2nd columns of the restoration polygon's attribute table contains these"
            " years."
        )
    if disturbance_start is not None:
        disturbance_start = str(disturbance_start)
    if restoration_start is not None:
        restoration_start = str(restoration_start)
    if reference_year_start is not None and reference_year_end is not None:
        reference_years = [str(reference_year_start), str(reference_year_end)]

    return disturbance_start, restoration_start, reference_years
    

def _validate_dates(rest_frame, ref_frame, image_stack):
    (
        disturbance_start,
        restoration_start,
        reference_years,
    ) = _get_dates_from_frame(
        rest_frame=rest_frame, ref_frame=ref_frame
    )

    if disturbance_start is None:
        raise ValueError(
            "disturbance start year missing (NULL provided)" 
        )
    if restoration_start is None:
        raise ValueError(
            "restoration start year missing (NULL provided)"
        )

    validate_year_format(disturbance_start, "disturbance start")
    validate_year_format(restoration_start, "restoration start")
    validate_year_order(disturbance_start, restoration_start)

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
        raise ValueError(
            "restoration_polygon is not within the bounds of images"
        ) from None

    return restoration_polygon


def _validate_reference_polygons(reference_polygons, image_stack):
    if reference_polygons is not None:
        if not image_stack.satts.contains_spatial(reference_polygons):
            raise ValueError(
                "not all reference_polygons within the bounds of images"
            ) from None

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
        if composite_stack.satts.is_annual_composite:
            self.restoration_polygon = _validate_restoration_polygons(
                restoration_polygon=restoration_polygon, image_stack=composite_stack
            )
            
            (
                self.reference_years,
                self.disturbance_start,
                self.restoration_start,
            ) = _validate_dates(
                rest_frame=restoration_polygon,
                ref_frame=reference_polygons,
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

        self.reference_polygons = _validate_reference_polygons(
            reference_polygons=reference_polygons, image_stack=composite_stack
        )
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

    def y2r(self, timestep = 5, percent_of_target: int = 80):
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
        y2r = y2r.expand_dims(dim={"metric": [str(Metric.Y2R)]})
        return y2r

    def yryr(self, timestep: int = 5, percent_of_target: int = 80):
        """Compute the Relative Years to Recovery (YRYR) metric."""
        yryr = m.yryr(
            image_stack=self.stack,
            rest_start=self.restoration_start,
            timestep=timestep,
        )
        yryr = yryr.expand_dims(dim={"metric": [str(Metric.YRYR)]})
        return yryr

    def dnbr(self, timestep: int = 5, percent_of_target: int = 80):
        """Compute the differenced normalized burn ratio (dNBR) metric."""
        dnbr = m.dnbr(
            image_stack=self.stack,
            rest_start=self.restoration_start,
            timestep=timestep,
        )
        dnbr = dnbr.expand_dims(dim={"metric": [str(Metric.DNBR)]})
        return dnbr

    def _rri(self, timestep: int = 5, percent_of_target: int = 80):
        """Compute the relative recovery index (RRI) metric."""
        rri = m.rri(
            image_stack=self.stack,
            rest_start=self.restoration_start,
            dist_start=self.disturbance_start,
            timestep=timestep,
        )
        rri = rri.expand_dims(dim={"metric": [str(Metric.RRI)]})
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
        r80p = r80p.expand_dims(dim={"metric": [str(Metric.R80P)]})
        return r80p

    def plot_spectral_trajectory(self, path: str = None, legend: bool = True) -> None:
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

        # convert stats xarray and recovery target xarray into merged df for plotting
        stats = stats.assign_coords(band=([str(b) for b in stats.band.values]))
        stats = stats.to_dataframe("value")

        recovery_target = self.recovery_target.assign_coords(
            band=([str(b) for b in self.recovery_target.band.values])
        )
        reco_targets = recovery_target.to_dataframe("reco_targets").dropna(how="any")

        # merge on multi-index: (statistic, band, year) then reset index
        data = stats.merge(reco_targets, left_index=True, right_index=True)[
            ["value", "reco_targets"]
        ]
        data = data.reset_index()
        data["time"] = data["time"].apply(lambda x: str(x.year))

        # Set theme and colour palette for plots
        sns.set_theme()
        palette = sns.color_palette("deep")

        bands = data["band"].unique()
        fig, axs = plt.subplots(
            len(bands), 1, sharey=False, sharex=True, figsize=[8, 7.5]
        )
        # Plot per-band statistic lineplots
        for i, band in enumerate(bands):
            band_data = data[data["band"] == band]
            try:
                axi = axs[i]
            except TypeError:
                axi = axs

            sns.lineplot(
                data=band_data,
                x="time",
                hue="stats",
                y="value",
                ax=axi,
                legend=False,
                lw=1,
            )
            sns.lineplot(
                data=band_data[band_data["stats"] == "mean"],
                x="time",
                y="reco_targets",
                ax=axi,
                color="black",
                linestyle=(0, (3, 5, 1, 5)),
                lw=1,
            )
            self._draw_trajectory_windows(axi, palette, hist_ref_sys)
            self._set_axis_labels(axi, band, data["time"].unique().tolist())
        (
            labels,
            custom_handles,
        ) = self._custom_legend_labels_handles(palette, hist_ref_sys)

        if legend:
            plt.figlegend(
                labels=labels,
                handles=custom_handles,
                loc="lower center",
                fancybox=True,
                ncol=3,
                handler_map={Patch: HandlerFilledBetween()},
            )
            plt.subplots_adjust(
                bottom=plt.rcParams["figure.subplot.bottom"]
                + (plt.rcParams["figure.subplot.bottom"] / 1.5)
            )
        if path:
            plt.savefig(path)
        else:
            plt.show()

    def _set_axis_labels(self, axi, title, xlabels):
        """Set the axis labels to desired values"""
        axi.set_xticks(
            axi.get_xticks(),
            xlabels,
            rotation=45,
            ha="right",
        )
        axi.set_xlabel("Year")
        axi.set_ylabel(f"{title} Value")

    def _draw_trajectory_windows(self, axi, palette, hist_ref_sys):
        """Draw the trajectory windows onto subplots.

        Uses two verticle dashed lines to delimit the start and
        end years of a window. If the start and end years are
        not the same year, then the space between the two dashed lines
        is filled in (vertical span). Each window (i.e line/span group)
        is coloured a distinct colour.

        Draws the reference, disturbance, and recovery windows.

        """
        # Draw recovery window
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
            x=str(self.end_year.year),
            color=palette[2],
            linestyle="dashed",
            lw=1,
        )

        # Draw disturbance window
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
            # if deriving target from recovery polygon, draw reference window
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
            # only draw line if reference ye
            if self.reference_years[1] != self.disturbance_start:
                axi.axvline(
                    x=self.reference_years[1],
                    color=palette[4],
                    linestyle="dashed",
                    lw=1,
                )

    def _custom_legend_labels_handles(self, palette, hist_ref_sys) -> Tuple[List, List]:
        """Create a custom legend to match trajectory plots

        Returns
        -------
        tuple of lists
            custom labels and handles to pass to ``figlegend``

        """
        median_line = Line2D([0], [0], color=palette[0], lw=2)
        mean_line = Line2D([0], [0], color=palette[1], lw=2)
        recovery_target_line = Line2D(
            [0], [0], color="black", linestyle=(0, (3, 5, 1, 5)), lw=1
        )
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
    """Custom Patch Handler for trajectory windows.

    Draws Patch objects with left and right edges coloured/dashed
    to match the style of trajectory window Patches in the plots.

    """

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        p = super().create_artists(
            legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
        )[0]
        color = p.get_facecolor()
        x0, y0 = 0, 0
        x1 = x0 + width
        y1 = y0 + height
        line_left = Line2D(
            [x0, x0], [y0, y1], color=color, linestyle="dashed", lw=0.85, alpha=1
        )
        line_right = Line2D(
            [x1, x1], [y0, y1], color=color, linestyle="dashed", lw=0.85, alpha=1
        )
        return [p, line_left, line_right]
