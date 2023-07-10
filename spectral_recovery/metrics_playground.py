import xarray as xr
import numpy as np
from utils import maintain_spatial_attrs
from sklearn.linear_model import TheilSenRegressor
import timeit
from scipy import stats


@maintain_spatial_attrs
def years_to_recovery(
    image_stack: xr.DataArray,
    baseline: xr.DataArray,
    percent: int = 80,
) -> xr.DataArray:
    """Per-pixel years-to-recovery

    Parameters
    ----------
    image_stack : xr.DataArray
        Timeseries of images to compute years-to-recovery across.

    """
    reco_80 = baseline * (percent / 100)
    # theil_sen calls apply_ufunc along the time dimension so stack's
    # chunks need to contain the entire timestack before being passed
    y_vals = image_stack.chunk(dict(time=-1))
    x_vals = image_stack.time.dt.year

    ts = theil_sen(y=y_vals, x=x_vals)
    y2r = (reco_80 - ts.sel(parameter="intercept")) / ts.sel(parameter="slope")
    # TODO: maybe return NaN if intercept + slope*curr_year is not recovered
    return y2r - x_vals[0]


def new_linregress(y, x):
    """Wrapper around mstats.theilslopes for apply_ufunc usage"""
    slope, intercept, low_slope, high_slope = stats.mstats.theilslopes(y, x)
    return np.array([slope, intercept])


def theil_sen(y, x):
    """Apply theil_sen slope regression across time on each pixel

    Parameters
    ----------
    y : xr.DataArray

    x : list of int

    Returns
    -------
    ts_reg : xr.DataArray
        DataArray of  theil-sen slope and intercept parameters for each
        pixel. 3D DataArray with "parameter", "y" and "x" labelled
        dimensions where "y" and "x" match input "y" and "x".

    """
    ts_dim_name = "parameter"
    ts_reg = xr.apply_ufunc(
        new_linregress,
        y,
        x,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[ts_dim_name]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float64"],
        dask_gufunc_kwargs={"output_sizes": {ts_dim_name: 2}},
    )
    ts_reg = ts_reg.assign_coords({"parameter": ["slope", "intercept"]})
    return ts_reg


def _theilsen_regression(pixel):
    """
    Parameters
    ----------
    pixel : pd.Series
        Timeseries of pixel

    Returns
    -------
    pd.Dataframe
        Slope and intercept vals

    """
    slope, intercept, low_slope, high_slope = stats.mstats.theilslopes(pixel)
    return pd.DataFrame({"slope": [slope], "intercept": [intercept]})


def _trend_block_mapper(block):
    """Determine per-pixel, per-band trajectory params across time.

    Parameters
    ----------
    block : np.array
        4D array (band, time, y, x) of timeseries data.

    Returns
    -------
    block_with_trajectory : np.array
        4D array (band, time, y, x) containing per-pixel, per-band trajectory
        params along the time dimension.

    """
    block_without_time = (block.shape[0], block.shape[2], block.shape[3])
    block_as_df = pd.DataFrame(
        np.column_stack(
            list(
                map(np.ravel, np.meshgrid(*map(np.arange, block.shape), indexing="ij"))
            )
            + [block.ravel()]
        ),
        columns=["band", "time", "y", "x", "val"],
    )  # Fast. From https://stackoverflow.com/questions/45422898
    reg_out = (
        block_as_df.groupby(["x", "y", "band"], group_keys=True)["val"]
        .apply(_theilsen_regression)  # TODO: parameterize this
        .reset_index()
    )
    indices = reg_out.index.tolist()

    params = ["slope", "intercept"]
    param_arrays = []
    for param in params:
        # Move dataframe values back into array/block format
        val = reg_out[param].values.tolist()
        val_array = np.ones(block_without_time).flatten()
        val_array[indices] = val
        param_arrays.append(val_array.reshape(*block_without_time))
    # Deal with map_blocks requirement that axis be the same size
    filler = np.ones(block_without_time)
    filled = np.repeat([filler], block.shape[1] - 2).reshape(
        block.shape[0], block.shape[1] - len(params), block.shape[2], block.shape[3]
    )

    block_with_trajectory = np.stack(tuple(param_arrays), axis=1)
    block_with_trajectory = np.concatenate((block_with_trajectory, filled), axis=1)

    return block_with_trajectory


def get_trajectory(array):
    """Coordinator for calling regression function over Dask blocks.

    TODO: this function could be maybe be redundant if we find a way to let map_block
    calls have a new dimension on return (e.g params, band, y, x) vs. (band, time, y, x)
    """
    # TODO: Chunk sizes should be dynamic not hard-coded
    array = array.chunk((1, -1, 10, 10))
    array.data = array.data.map_blocks(_trend_block_mapper, dtype=np.ndarray)

    array = array[:, 0:2, :, :].rename({"time": "parameter"})
    array = array.assign_coords({"parameter": ["slope", "intercept"]})

    return array


@maintain_spatial_attrs
def simple_years_to_recovery(
    image_stack: xr.DataArray,
    baseline: xr.DataArray,
    percent: int = 80,
) -> xr.DataArray:
    """Per-pixel years-to-recovery

    Parameters
    ----------
    image_stack : xr.DataArray
        Timeseries of images to compute years-to-recovery across.
    # TODO: predictive vs. non-predictive
    """
    reco_80 = baseline * (percent / 100)
    # NOTE: eventually, this call should allow "type of trajectory/regression" as a param
    # That way we can seperate the Y2R equation from trajectory algos
    ts = get_trajectory(image_stack)
    # print(ts.sel(parameter="slope").data.compute().shape)
    y2r = (reco_80 - ts.sel(parameter="intercept")) / ts.sel(parameter="slope")
    return y2r


if __name__ == "__main__":
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster()  # Launches a scheduler and workers locally
    client = Client(cluster)

    import dask.array as da
    import pandas as pd
    import rioxarray

    array = xr.DataArray(
        da.from_array(
            np.arange(0, 500 * 500 * 5).reshape(1, 5, 500, 500), chunks="auto"
        ),
        coords=[
            ("band", ["red"]),
            (
                "time",
                [
                    pd.to_datetime("2010"),
                    pd.to_datetime("2011"),
                    pd.to_datetime("2012"),
                    pd.to_datetime("2013"),
                    pd.to_datetime("2014"),
                ],
            ),
            ("y", np.arange(0, 500)),
            ("x", np.arange(0, 500)),
        ],
    )
    # TODO (Friday) the baseline dimensions don't match what is happening in spectral_recovery --- fix this then see if the output from simple_years_to_recovery has appropriate dimensions
    # Something about the time dimensions
    bline = xr.DataArray(
        da.from_array(np.ones(shape=(1, 1, 1, 1)) * 20, chunks="auto"),
        coords=[("band", ["red"]), ("time", [0]), ("y", [0]), ("x", [0])],
    )
    array = array.rio.write_crs("4326")
    bline = bline.rio.write_crs("4326")

    # array = array.data.compute()
    # start1 = timeit.default_timer()
    # df = pd.DataFrame(np.column_stack(list(map(np.ravel,
    #                                            np.meshgrid(*map(np.arange, array.shape), indexing="ij"))) + [array.ravel()]),
    #                                            columns=["band", "time", "y", "x", "val"])
    # sloped_df = df.groupby(["x","y","band"], group_keys=True)["val"].apply(theilsen_trend)
    # stop1 = timeit.default_timer()
    # print(f"pd: {stop1-start1}")
    # print(sloped_df)

    # print("Starting apply_ufunc...")
    # start1 = timeit.default_timer()
    # years_to_recovery(array, bline).compute()
    # stop1 = timeit.default_timer()
    # print(f"apply_ufunc: {stop1-start1}")
    # print("Done apply_ufunc")

    # print("Starting Barry's")
    # start2 = timeit.default_timer()
    # TheilSen_regression(array.sel(band="red"), client, debug=False)
    # TheilSen_regression(array.sel(band="blue"), client, debug=False)
    # TheilSen_regression(array.sel(band="green"), client, debug=False)
    # stop2 = timeit.default_timer()
    # print(f"barry's: {stop2-start2}")

    print("Starting Simple")
    start3 = timeit.default_timer()
    res = simple_years_to_recovery(array, bline)
    print(res.data.compute())
    stop3 = timeit.default_timer()
    print(f"simple's: {stop3-start3}")
