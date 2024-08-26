"""Methods for plotting spectral trajectories"""

from typing import List, Dict, Tuple

import geopandas as gpd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerPatch

from spectral_recovery.timeseries import _SatelliteTimeSeries


def plot_spectral_trajectory(
    timeseries_data: xr.DataArray,
    restoration_site: gpd.GeoDataFrame,
    recovery_target: xr.DataArray,
    reference_start: int = None,
    reference_end: int = None,
    path: str = None,
):
    """Plot the spectral trajectory of the restoration polygon

    Parameters
    ----------
    timeseries_data : xarray.DataArray
        A timeseries of indices to plot the spectral trajectory with.
    restoration_site : gpd.GeoDataFrame
        The resoration site to plot a spectral trajectory of.
    recovery_target : xarray.DataArray
        The recovery target value for the given restoration site.
    reference_start : int
        The first/start reference year.
    reference_end : int
        The final/end reference year.
    path : str
        Path and filename for writing plot.

    """
    clipped_timeseries = timeseries_data.rio.clip(restoration_site.geometry.values)
    _plot_ra(
        disturbance_start=str(restoration_site["dist_start"].iloc[0]),
        restoration_start=str(restoration_site["rest_start"].iloc[0]),
        timeseries_data=clipped_timeseries,
        recovery_target=recovery_target,
        reference_start=str(reference_start),
        reference_end=str(reference_end),
        path=path,
    )


def _plot_ra(
    disturbance_start: int,
    restoration_start: int,
    timeseries_data: xr.DataArray,
    recovery_target: xr.DataArray,
    reference_start: str,
    reference_end: str,
    path: str = None,
    legend: bool = True,
    figsize: list = None,
) -> None:
    """Create spectral trajectory plot of the RestorationArea (ra)

    Parameters
    ----------
    path : str, optional
        The path to save the plot to.
    """

    plot_ref_window = bool(reference_start and reference_end)

    stats = timeseries_data.satts.stats()
    stats = stats.sel(
        stats=[
            "median",
            "mean",
        ]
    )

    # convert stats xarray and recovery target xarray into merged df for plotting
    stats = stats.assign_coords(band=([str(b) for b in stats.band.values]))
    stats = stats.to_dataframe("value")

    recovery_target = recovery_target.assign_coords(
        band=([str(b) for b in recovery_target.band.values])
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
    fig, axs = plt.subplots(len(bands), 1, sharey=False, sharex=True, figsize=[8, 7.5])
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
        timeseries_end = str(np.max(timeseries_data["time"].dt.year.values))
        _draw_trajectory_windows(
            restoration_start,
            disturbance_start,
            timeseries_end,
            reference_start,
            reference_end,
            axi,
            palette,
            plot_ref_window,
        )
        _set_axis_labels(axi, band, data["time"].unique().tolist())
    (
        labels,
        custom_handles,
    ) = _custom_legend_labels_handles(palette, plot_ref_window)

    if legend:
        plt.figlegend(
            labels=labels,
            handles=custom_handles,
            loc="lower center",
            fancybox=True,
            ncol=3,
            handler_map={Patch: _HandlerFilledBetween()},
        )
        plt.subplots_adjust(
            bottom=plt.rcParams["figure.subplot.bottom"]
            + (plt.rcParams["figure.subplot.bottom"] / 1.5)
        )
    if path:
        plt.savefig(path)
    else:
        plt.show()


def _set_axis_labels(axi, title, xlabels):
    """Set the axis labels to desired values"""
    axi.set_xticks(
        axi.get_xticks(),
        xlabels,
        rotation=45,
        ha="right",
    )
    axi.set_xlabel("Year")
    axi.set_ylabel(f"{title} Value")


def _draw_trajectory_windows(
    restoration_start,
    disturbance_start,
    timeseries_end,
    refs,
    refe,
    axi,
    palette,
    plot_ref_window,
):
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
        x=restoration_start,
        color=palette[2],
        linestyle="dashed",
        lw=1,
    )
    axi.axvspan(
        restoration_start,
        timeseries_end,
        alpha=0.2,
        color=palette[2],
    )
    axi.axvline(
        x=timeseries_end,
        color=palette[2],
        linestyle="dashed",
        lw=1,
    )

    # Draw disturbance window
    axi.axvline(
        x=disturbance_start,
        color=palette[3],
        linestyle="dashed",
        lw=1,
    )
    axi.axvspan(
        disturbance_start,
        restoration_start,
        alpha=0.2,
        color=palette[3],
    )

    if plot_ref_window:
        # if deriving target from recovery polygon, draw reference window
        axi.axvline(
            x=refs,
            color=palette[4],
            linestyle="dashed",
            lw=1,
        )
        axi.axvspan(
            refs,
            refe,
            alpha=0.2,
            color=palette[4],
        )
        # only draw line if reference ye
        if refe != refs:
            axi.axvline(
                x=refe,
                color=palette[4],
                linestyle="dashed",
                lw=1,
            )


def _custom_legend_labels_handles(palette, plot_ref_window) -> Tuple[List, List]:
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
    if plot_ref_window:
        custom_handles.insert(3, reference_years_patch)
        labels.insert(3, "reference year(s)")
    custom_handles.insert(
        2,
        recovery_target_line,
    )
    labels.insert(2, "recovery target")

    return labels, custom_handles


class _HandlerFilledBetween(HandlerPatch):
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
