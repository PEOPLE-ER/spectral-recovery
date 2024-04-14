from typing import List, Dict, Tuple

import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerPatch

from spectral_recovery.restoration import RestorationArea
from spectral_recovery.indices import compute_indices


# TODO: Refactor. Bring plot_spectral_trajectory into this module.
def plot_spectral_trajectory(
    timeseries_data: xr.DataArray,
    restoration_polygons: gpd.GeoDataFrame,
    recovery_target: xr.DataArray,
    path: str = None,
):
    """Plot the spectral trajectory of the restoration polygon

    Parameters
    ----------
    timeseries_data : xr.DataArray
        Timeseries data (annual composites)
    restoration_polygons : gpd.GeoDataFrame
        Restoration polygon and dates
    indices : list of str
        Indices to visualize trajectory for
    reference_polygons : gpd.GeoDataFrame, optional
        The refernce polygons to compute recovery target with
    indices_constants : dict
        Constant values for indices
    recovery_target_method : callable
        Recovery target method to derive recovery target values

    """
    restoration_area = RestorationArea(
        restoration_polygon=restoration_polygons,
        composite_stack=timeseries_data,
        recovery_target_method=recovery_target,
    )
    if path:
        plot_ra(ra=restoration_area, path=path)
    else:
        plot_ra(ra=restoration_area)


def plot_ra(ra: RestorationArea, path: str = None, legend: bool = True) -> None:
    """Create spectral trajectory plot of the RestorationArea (ra)

    Parameters
    ----------
    ra : spectral_recovery.restoration.RestorationArea
        The restoration area to plot.
    path : str, optional
        The path to save the plot to.
    """
    hist_ref_sys = ra.reference_polygons is None

    stats = ra.restoration_image_stack.satts.stats()
    stats = stats.sel(
        stats=[
            "median",
            "mean",
        ]
    )

    # convert stats xarray and recovery target xarray into merged df for plotting
    stats = stats.assign_coords(band=([str(b) for b in stats.band.values]))
    stats = stats.to_dataframe("value")

    recovery_target = ra.recovery_target.assign_coords(
        band=([str(b) for b in ra.recovery_target.band.values])
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
        _draw_trajectory_windows(ra, axi, palette, hist_ref_sys)
        _set_axis_labels(ra, axi, band, data["time"].unique().tolist())
    (
        labels,
        custom_handles,
    ) = _custom_legend_labels_handles(ra, palette, hist_ref_sys)

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


def _draw_trajectory_windows(ra, axi, palette, hist_ref_sys):
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
        x=ra.restoration_start,
        color=palette[2],
        linestyle="dashed",
        lw=1,
    )
    axi.axvspan(
        ra.restoration_start,
        ra.timeseries_end,
        alpha=0.2,
        color=palette[2],
    )
    axi.axvline(
        x=ra.timeseries_end,
        color=palette[2],
        linestyle="dashed",
        lw=1,
    )

    # Draw disturbance window
    axi.axvline(
        x=ra.disturbance_start,
        color=palette[3],
        linestyle="dashed",
        lw=1,
    )
    axi.axvspan(
        ra.disturbance_start,
        ra.restoration_start,
        alpha=0.2,
        color=palette[3],
    )

    if hist_ref_sys:
        # if deriving target from recovery polygon, draw reference window
        axi.axvline(
            x=ra.reference_years[0],
            color=palette[4],
            linestyle="dashed",
            lw=1,
        )
        axi.axvspan(
            ra.reference_years[0],
            ra.reference_years[1],
            alpha=0.2,
            color=palette[4],
        )
        # only draw line if reference ye
        if ra.reference_years[1] != ra.disturbance_start:
            axi.axvline(
                x=ra.reference_years[1],
                color=palette[4],
                linestyle="dashed",
                lw=1,
            )


def _custom_legend_labels_handles(ra, palette, hist_ref_sys) -> Tuple[List, List]:
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
        if isinstance(ra.recovery_target_method, MedianTarget):
            if ra.recovery_target_method.scale == "pixel":
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
