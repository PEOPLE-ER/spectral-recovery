"""Methods for computing recovery metrics."""

from typing import Dict, List, Optional, Tuple
import warnings

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

from spectral_recovery.utils import maintain_rio_attrs

warnings.filterwarnings(
    "ignore", message="invalid value encountered in divide", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore", message="All-NaN slice encountered", category=RuntimeWarning
)

NEG_TIMESTEP_MSG = "timestep cannot be negative."
VALID_PERC_MSP = "percent must be between 0 and 100."
VALID_VALUE_SOURCE_MSG = "value_source must be either 'observed' or 'sen_slope'."
SEN_MIN_VALID_POINTS_MSG = "sen_min_valid_points must be >= 1."
VALUE_SOURCE_OBSERVED = "observed"
VALUE_SOURCE_SEN_SLOPE = "sen_slope"
METRIC_FUNCS = {}


def _register_metrics(f):
    """Add function and name to global name/func dict"""
    METRIC_FUNCS[f.__name__] = f
    return f


def _validate_value_source(value_source: str) -> str:
    """Validate metric value retrieval mode."""
    normalized = str(value_source).lower()
    if normalized not in {VALUE_SOURCE_OBSERVED, VALUE_SOURCE_SEN_SLOPE}:
        raise ValueError(VALID_VALUE_SOURCE_MSG)
    return normalized


def _validate_sen_min_valid_points(sen_min_valid_points: int) -> int:
    """Validate minimum number of finite points for Sen-slope fitting."""
    try:
        parsed = int(sen_min_valid_points)
    except (TypeError, ValueError):
        raise ValueError(SEN_MIN_VALID_POINTS_MSG) from None

    if parsed < 1:
        raise ValueError(SEN_MIN_VALID_POINTS_MSG)
    return parsed


def _timeseries_year_bounds(timeseries_data: xr.DataArray) -> Tuple[int, int]:
    """Get first and last year represented by timeseries_data."""
    start_year = int(timeseries_data.time.dt.year.min().item())
    end_year = int(timeseries_data.time.dt.year.max().item())
    return start_year, end_year


def _template_nan_like(timeseries_data: xr.DataArray) -> xr.DataArray:
    """Return a band/y/x template filled with NaN values."""
    template = timeseries_data.isel(time=0).drop_vars("time")
    return xr.full_like(template, np.nan, dtype=np.float64)


def _observed_year_value(timeseries_data: xr.DataArray, year: int) -> xr.DataArray:
    """Return observed band/y/x values for a given year or NaN template if absent."""
    year_dt = pd.to_datetime(str(year))
    if year_dt not in timeseries_data.time.values:
        return _template_nan_like(timeseries_data)

    observed = timeseries_data.sel(time=str(year)).drop_vars("time")
    try:
        observed = observed.squeeze("time")
    except (KeyError, ValueError):
        pass
    return observed


def _sen_slope_1d_fit(
    pixel_values: np.ndarray,
    years: np.ndarray,
    sen_min_valid_points: int,
) -> Tuple[float, float]:
    """Fit one pixel's Sen-slope model and return slope/intercept."""
    finite = np.isfinite(pixel_values)
    if not np.any(finite):
        return np.nan, np.nan

    obs_vals = pixel_values[finite].astype(np.float64)
    obs_years = years[finite].astype(np.float64)

    if obs_vals.size == 1:
        return 0.0, float(obs_vals[0])

    if obs_vals.size < sen_min_valid_points:
        return np.nan, np.nan

    slopes = []
    for i in range(obs_vals.size - 1):
        deltas_year = obs_years[i + 1 :] - obs_years[i]
        valid = deltas_year != 0
        if not np.any(valid):
            continue
        deltas_val = obs_vals[i + 1 :] - obs_vals[i]
        slopes.append(deltas_val[valid] / deltas_year[valid])

    if not slopes:
        return 0.0, float(obs_vals[-1])

    slope = float(np.nanmedian(np.concatenate(slopes)))
    if np.isnan(slope):
        return np.nan, np.nan

    intercept = float(np.nanmedian(obs_vals - (slope * obs_years)))
    return slope, intercept


def _sen_slope_fit_per_pixel(
    timeseries_data: xr.DataArray,
    sen_min_valid_points: int,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute per-pixel Sen-slope fit arrays for slope and intercept."""
    modeled_input = timeseries_data.astype(np.float64)
    # apply_ufunc with a core dim requires a single chunk along that dim.
    # Years are typically short, so rechunking time is safe and avoids runtime errors.
    if hasattr(modeled_input.data, "chunks"):
        modeled_input = modeled_input.chunk({"time": -1})

    years = modeled_input.coords["time"].dt.year.values.astype(np.float64)
    slope_da, intercept_da = xr.apply_ufunc(
        _sen_slope_1d_fit,
        modeled_input,
        input_core_dims=[["time"]],
        output_core_dims=[[], []],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.float64, np.float64],
        kwargs={
            "years": years,
            "sen_min_valid_points": sen_min_valid_points,
        },
    )
    return slope_da, intercept_da


def _modeled_value_from_fit(
    slope_da: xr.DataArray,
    intercept_da: xr.DataArray,
    year: int,
) -> xr.DataArray:
    """Predict per-pixel values at a target year from cached fit arrays."""
    return intercept_da + (slope_da * float(year))


def _metric_year_value(
    timeseries_data: xr.DataArray,
    year: int,
    value_source: str,
    sen_min_valid_points: int,
    sen_slope_fit_slope: Optional[xr.DataArray] = None,
    sen_slope_fit_intercept: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Get per-pixel value at year from observed or Sen-slope modeled source."""
    source = _validate_value_source(value_source)
    min_points = _validate_sen_min_valid_points(sen_min_valid_points)

    observed = _observed_year_value(timeseries_data=timeseries_data, year=year)
    if source == VALUE_SOURCE_OBSERVED:
        return observed

    if sen_slope_fit_slope is None or sen_slope_fit_intercept is None:
        sen_slope_fit_slope, sen_slope_fit_intercept = _sen_slope_fit_per_pixel(
            timeseries_data=timeseries_data,
            sen_min_valid_points=min_points,
        )

    modeled = _modeled_value_from_fit(
        slope_da=sen_slope_fit_slope,
        intercept_da=sen_slope_fit_intercept,
        year=year,
    )
    return observed.where(~observed.isnull(), modeled)


@maintain_rio_attrs
def compute_metrics(
    metrics: List[str],
    timeseries_data: xr.DataArray,
    restoration_sites: gpd.GeoDataFrame,
    recovery_targets: xr.DataArray | Dict = None,
    timestep: int = 5,
    percent_of_target: int = 80,
    value_source: str = VALUE_SOURCE_OBSERVED,
    sen_min_valid_points: int = 2,
) -> Dict:
    """Compute recovery metrics for each restoration site.

    Parameters
    ----------
    metrics : list of str
        The names of recovery metrics to compute. Accepted values:
            - "Y2R": Years-to-Recovery
            - "R80P": Recovered 80 Percent
            - "deltaIR": delta Index Regrowth
            - "YrYr": Year-on-Year Average
            - "RRI": Relative Recovery Indicator
    timeseries_data : xarray.DataArray
        The timeseries of indices to compute recovery metrics with.
        Must contain band, time, y, and x dimensions.
    restoration_sites : geopandas.GeoDataFrame
        The restoration sites to compute a recovery targets for.
    recovery_targets : xarray.DataArray or dict
        The recovery targets. Either a dict mapping polygon IDs to
        xarray.DataArrays of recovery targets or a single xarray.DataArray.
    timestep : int, optional
        The timestep post-restoration to consider when computing recovery
        metrics. Only used for "R80P", "deltaIR", and "YrYr" and "RRI" recovery
        metrics. Default = 5.
    percent_of_target : int, optional
        The percent of the recovery target to consider when computing
        recovery metrics. Only used for "Y2R" and "R80P". Default = 80.
    value_source : str, optional
        Source for required yearly metric values. "observed" uses only
        observed values at each required year. "sen_slope" fills missing
        required-year values with a per-pixel Sen-slope modeled value.
        Default = "observed".
    sen_min_valid_points : int, optional
        Minimum finite yearly observations required to fit Sen-slope.
        If only one finite observation exists for a pixel, a constant
        fallback is used. Default = 2.

    Returns
    -------
    metric_ds : xarray.Dataset
        Dataset of restoration site ID variables, each containing an array
        of recovery metrics specific to each site.

    Notes
    -----
    Recovery target arrays _must_ be broadcastable to the timeseries_data
    when timeseries_data is clipped to each restoration site.

    """
    if recovery_targets is None:
        for tmetric in ["Y2R", "R80P"]:
            if tmetric in metrics:
                raise ValueError(
                    f"{tmetric} requires a recovery target but recovery_target is None"
                )
    value_source = _validate_value_source(value_source)
    sen_min_valid_points = _validate_sen_min_valid_points(sen_min_valid_points)

    per_polygon_metrics = {}
    for site_id, row in restoration_sites.iterrows():
        # Prepare arguments being passed to the metric functions
        clipped_ts = timeseries_data.rio.clip([row.geometry])
        sen_slope_fit_slope = None
        sen_slope_fit_intercept = None
        if value_source == VALUE_SOURCE_SEN_SLOPE:
            sen_slope_fit_slope, sen_slope_fit_intercept = _sen_slope_fit_per_pixel(
                timeseries_data=clipped_ts,
                sen_min_valid_points=sen_min_valid_points,
            )

        all_kwargs = {
            "disturbance_start": row["dist_start"],
            "restoration_start": row["rest_start"],
            "timeseries_data": clipped_ts,
            "timestep": timestep,
            "percent_of_target": percent_of_target,
            "value_source": value_source,
            "sen_min_valid_points": sen_min_valid_points,
            "sen_slope_fit_slope": sen_slope_fit_slope,
            "sen_slope_fit_intercept": sen_slope_fit_intercept,
        }
        if isinstance(recovery_targets, dict):
            all_kwargs["recovery_target"] = recovery_targets[site_id]
        else:
            # if a DataArray or None, just pass as-is
            all_kwargs["recovery_target"] = recovery_targets
        m_results = []
        for m in metrics:
            try:
                m_func = METRIC_FUNCS[m.lower()]
            except KeyError:
                raise ValueError(f"{m} is not a valid metric choice!") from None
            func_kwargs = {
                k: all_kwargs[k]
                for k in m_func.__code__.co_varnames
                if k in list(all_kwargs.keys())
            }
            m_results.append(m_func(**func_kwargs).assign_coords({"metric": m}))
        per_polygon_metrics[site_id] = xr.concat(m_results, "metric")

    return per_polygon_metrics


def _has_continuous_years(images: xr.DataArray):
    """Check for continous set of years in DataArray"""
    years = images.coords["time"].dt.year.values
    for year in list(range(years[0], years[-1] + 1)):
        if year not in years:
            return False
    return True


@_register_metrics
def deltair(
    restoration_start: int,
    timeseries_data: xr.DataArray,
    timestep: int = 5,
    value_source: str = VALUE_SOURCE_OBSERVED,
    sen_min_valid_points: int = 2,
    sen_slope_fit_slope: Optional[xr.DataArray] = None,
    sen_slope_fit_intercept: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Per-pixel deltaIR.

    The absolute change in a spectral index’s value at a point in the
    restoration monitoring window from the start of the restoration monitoring
    window. The default is the change that has occurred 5 years into the
    restoration from the start of the restoration.

    Parameters
    ----------
    restoration_start : int
        The start year of restoration activities.
    timeseries_data:
        The timeseries of indices to compute dIR with. Must contain
        band, time, y, and x coordinate dimensions.
    timestep : int
        The timestep post-restoration to compute deltaIR with.

    Returns
    -------
    deltair_v : xr.DataArray
        DataArray containing the deltaIR value for each pixel.

    """
    if timestep < 0:
        raise ValueError(NEG_TIMESTEP_MSG)

    value_source = _validate_value_source(value_source)
    sen_min_valid_points = _validate_sen_min_valid_points(sen_min_valid_points)

    rest_post_t = restoration_start + timestep
    _, timeseries_end = _timeseries_year_bounds(timeseries_data)
    if int(rest_post_t) > int(timeseries_end):
        raise ValueError(
            f" {restoration_start}+{timestep}={rest_post_t} is greater"
            f" than end of timeseries: {timeseries_end}. "
        ) from None

    deltair_v = _metric_year_value(
        timeseries_data=timeseries_data,
        year=rest_post_t,
        value_source=value_source,
        sen_min_valid_points=sen_min_valid_points,
        sen_slope_fit_slope=sen_slope_fit_slope,
        sen_slope_fit_intercept=sen_slope_fit_intercept,
    ) - _metric_year_value(
        timeseries_data=timeseries_data,
        year=restoration_start,
        value_source=value_source,
        sen_min_valid_points=sen_min_valid_points,
        sen_slope_fit_slope=sen_slope_fit_slope,
        sen_slope_fit_intercept=sen_slope_fit_intercept,
    )

    return deltair_v


@_register_metrics
def yryr(
    restoration_start: int,
    timeseries_data: xr.DataArray,
    timestep: int = 5,
    value_source: str = VALUE_SOURCE_OBSERVED,
    sen_min_valid_points: int = 2,
    sen_slope_fit_slope: Optional[xr.DataArray] = None,
    sen_slope_fit_intercept: Optional[xr.DataArray] = None,
):
    """Per-pixel YrYr.

    The average annual recovery rate relative to a fixed time interval
    during the restoration monitoring window. The default is the first 5
    years of the restoration window, however this can be changed by specifying
    the parameter `timestep`.

    Parameters
    ----------
    restoration_start : int
        The start year of restoration activities.
    timeseries_data:
        The timeseries of indices to compute YrYr with. Must contain
        band, time, y, and x coordinate dimensions.
    timestep : int
        The timestep post-restoration to compute YrYr with.

    Returns
    -------
    yryr_v : xr.DataArray
        DataArray containing the YrYr value for each pixel.

    """
    if timestep < 0:
        raise ValueError(NEG_TIMESTEP_MSG)

    value_source = _validate_value_source(value_source)
    sen_min_valid_points = _validate_sen_min_valid_points(sen_min_valid_points)

    rest_post_t = restoration_start + timestep
    _, timeseries_end = _timeseries_year_bounds(timeseries_data)
    if int(rest_post_t) > int(timeseries_end):
        raise ValueError(
            f" {restoration_start}+{timestep}={rest_post_t} is greater"
            f" than end of timeseries: {timeseries_end}. "
        ) from None

    obs_post_t = _metric_year_value(
        timeseries_data=timeseries_data,
        year=rest_post_t,
        value_source=value_source,
        sen_min_valid_points=sen_min_valid_points,
        sen_slope_fit_slope=sen_slope_fit_slope,
        sen_slope_fit_intercept=sen_slope_fit_intercept,
    )
    obs_start = _metric_year_value(
        timeseries_data=timeseries_data,
        year=restoration_start,
        value_source=value_source,
        sen_min_valid_points=sen_min_valid_points,
        sen_slope_fit_slope=sen_slope_fit_slope,
        sen_slope_fit_intercept=sen_slope_fit_intercept,
    )
    yryr_v = (obs_post_t - obs_start) / timestep
    return yryr_v


@_register_metrics
def r80p(
    restoration_start: int,
    timeseries_data: xr.DataArray,
    recovery_target: xr.DataArray,
    timestep: int = 5,
    percent_of_target: int = 80,
    value_source: str = VALUE_SOURCE_OBSERVED,
    sen_min_valid_points: int = 2,
    sen_slope_fit_slope: Optional[xr.DataArray] = None,
    sen_slope_fit_intercept: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Per-pixel R80P.

    The extent to which the trajectory has reached 80% of the recovery
    target value. The metric commonly uses the maximum value from the
    4th or 5th year of restoration window to show the extent to which a
    pixel has reached 80% of the target value 5 years into the restoration
    window. However for monitoring purposes, this tool uses the selected `timestep`
    or defaults to the current timestep to provide up to date recovery
    progress. 80% of the recovery target value is the default, however this
    can be changed by modifying the value of `percent`.

    Parameters
    ----------
    restoration_start : int
        The start year of restoration activities.
    timeseries_data:
        The timeseries of indices to compute R80P with. Must contain
        band, time, y, and x coordinate dimensions.
    recovery_target : xarray.DataArray
        Recovery target values. Must be broadcastable to timeseries_data.
    timestep : int
        The timestep post-restoration to compute R80P with.
    percent_of_target : int
        The percent of the recovery target to consider when computing
        R80P.

    Returns
    -------
    r80p_v : xr.DataArray
        DataArray containing the R80P value for each pixel.

    """
    if timestep < 0:
        raise ValueError(NEG_TIMESTEP_MSG)
    elif percent_of_target <= 0 or percent_of_target > 100:
        raise ValueError(VALID_PERC_MSP)
    else:
        rest_post_t = restoration_start + timestep

    value_source = _validate_value_source(value_source)
    sen_min_valid_points = _validate_sen_min_valid_points(sen_min_valid_points)

    _, timeseries_end = _timeseries_year_bounds(timeseries_data)
    if int(rest_post_t) > int(timeseries_end):
        raise ValueError(
            f" {restoration_start}+{timestep}={rest_post_t} is greater"
            f" than end of timeseries: {timeseries_end}. "
        ) from None

    r80p_v = _metric_year_value(
        timeseries_data=timeseries_data,
        year=rest_post_t,
        value_source=value_source,
        sen_min_valid_points=sen_min_valid_points,
        sen_slope_fit_slope=sen_slope_fit_slope,
        sen_slope_fit_intercept=sen_slope_fit_intercept,
    ) / ((percent_of_target / 100) * recovery_target)
    return r80p_v


@_register_metrics
def y2r(
    restoration_start: int,
    timeseries_data: xr.DataArray,
    recovery_target: xr.DataArray,
    percent_of_target: int = 80,
) -> xr.DataArray:
    """Per-pixel Y2R.

    The length of time taken (in time steps/years) for a given pixel to
    first reach 80% of its recovery target value. The percent can be modified
    by changing the value of `percent`.

    Parameters
    ----------
    restoration_start : int
        The start year of restoration activities.
    timeseries_data:
        The timeseries of indices to compute Y2R with. Must contain
        band, time, y, and x coordinate dimensions.
    recovery_target : xarray.DataArray
        Recovery target values. Must be broadcastable to timeseries_data.
    percent_of_target : int
        The percent of the recovery target to consider when computing
        Y2R.

    Returns
    -------
    y2r_v : xr.DataArray
        DataArray containing the number of years taken for each pixel
        to reach the recovery target value. NaN represents pixels that
        have not yet reached the recovery target value.

    """
    if percent_of_target <= 0 or percent_of_target > 100:
        raise ValueError(VALID_PERC_MSP)

    recovery_window = timeseries_data.sel(time=slice(str(restoration_start), None))
    if not _has_continuous_years(recovery_window):
        raise ValueError(
            f"Missing years in `timeseries_data`, cannot compute Y2R. Y2R requires a continuous timeseries from the restoration start year onwards."
        )
    y2r_target = recovery_target * (percent_of_target / 100)
    years_to_recovery = (recovery_window >= y2r_target).argmax(dim="time", skipna=True)
    # Pixels with value 0 could be:
    # 1. pixels that were recovered at the first timestep
    # 2. pixels that never recovered (argmax returns 0 if all values are False)
    # 3. pixels that were NaN for the entire recovery window.
    #
    # Only 1. is a valid 0, so set pixels that never recovered to -9999,
    # and pixels that were NaN for the entire recovery window back to NaN.
    not_zero = years_to_recovery != 0
    recovered_at_zero = recovery_window.sel(time=str(restoration_start)) >= y2r_target
    is_nan = recovery_window.isnull().all("time")
    valid_output = not_zero | recovered_at_zero | is_nan

    # Set unrecovered 0's to -9999, aund NaN 0's to NaN
    y2r_v = years_to_recovery.where(valid_output, -9999)
    y2r_v = y2r_v.where(~is_nan, np.nan).drop_vars("time")

    try:
        y2r_v = y2r_v.squeeze("time")
    except KeyError:
        pass
    return y2r_v


@_register_metrics
def rri(
    disturbance_start: int,
    restoration_start: int,
    timeseries_data: xr.DataArray,
    timestep: int = 5,
    value_source: str = VALUE_SOURCE_OBSERVED,
    sen_min_valid_points: int = 2,
    sen_slope_fit_slope: Optional[xr.DataArray] = None,
    sen_slope_fit_intercept: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Per-pixel RRI.

    A modified version of the commonly used RI, the RRI accounts for
    noise in trajectory by using the maximum from the 4th or 5th year
    in monitoring window. The metric relates recovery magnitude to
    disturbance magnitude, and is the change in index value in 4 or 5
    years divided by the change due to disturbance.

    Parameters
    ----------
    disturbance_start : int
        The start year of the disturbance event.
    restoration_start : int
        The start year of restoration activities.
    timeseries_data:
        The timeseries of indices to compute RRI with. Must contain
        band, time, y, and x coordinate dimensions.
    timestep : int
        The timestep post-restoration to compute RRI with.

    Returns
    -------
    rri_v : xr.DataArray
        DataArray containing the RRI value for each pixel.

    """
    if timestep < 0:
        raise ValueError(NEG_TIMESTEP_MSG)

    if timestep == 0:
        raise ValueError("timestep for RRI must be greater than 0.")

    value_source = _validate_value_source(value_source)
    sen_min_valid_points = _validate_sen_min_valid_points(sen_min_valid_points)

    rest_post_tm1 = restoration_start + (timestep - 1)
    rest_post_t = restoration_start + timestep
    _, timeseries_end = _timeseries_year_bounds(timeseries_data)
    if rest_post_t > timeseries_end:
        raise ValueError(
            f" {restoration_start}+{timestep}={rest_post_t} is greater"
            f" than end of timeseries: {timeseries_end}. "
        ) from None

    max_rest_t_tm1 = xr.concat(
        [
            _metric_year_value(
                timeseries_data=timeseries_data,
                year=rest_post_tm1,
                value_source=value_source,
                sen_min_valid_points=sen_min_valid_points,
                sen_slope_fit_slope=sen_slope_fit_slope,
                sen_slope_fit_intercept=sen_slope_fit_intercept,
            ),
            _metric_year_value(
                timeseries_data=timeseries_data,
                year=rest_post_t,
                value_source=value_source,
                sen_min_valid_points=sen_min_valid_points,
                sen_slope_fit_slope=sen_slope_fit_slope,
                sen_slope_fit_intercept=sen_slope_fit_intercept,
            ),
        ],
        dim=pd.Index([rest_post_tm1, rest_post_t], name="time"),
    ).max(dim="time", skipna=True)
    rest_start = _metric_year_value(
        timeseries_data=timeseries_data,
        year=restoration_start,
        value_source=value_source,
        sen_min_valid_points=sen_min_valid_points,
        sen_slope_fit_slope=sen_slope_fit_slope,
        sen_slope_fit_intercept=sen_slope_fit_intercept,
    )
    dist_start = _metric_year_value(
        timeseries_data=timeseries_data,
        year=disturbance_start,
        value_source=value_source,
        sen_min_valid_points=sen_min_valid_points,
        sen_slope_fit_slope=sen_slope_fit_slope,
        sen_slope_fit_intercept=sen_slope_fit_intercept,
    )
    dist_end = rest_start

    rri_v = (max_rest_t_tm1 - rest_start) / (dist_start - dist_end)
    return rri_v


def _year_dt(dt, dt_type: str = "int"):
    """Get int or str representation of year from datetime-like object."""
    # TODO: refuse to move forward if dt isn't datetime-like
    try:
        dt_dt = pd.to_datetime(dt)
        year = dt_dt.year
    except ValueError:
        raise ValueError(
            f"Unable to get year {type} from {dt} of type {type(dt)}"
        ) from None
    if dt_type == "str":
        return str(year)
    return year
