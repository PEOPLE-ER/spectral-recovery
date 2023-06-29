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


class TheilSen_regression():
        """
        Perform Theil-Sen robust regression through a timeseries image stack, per-pixel

        PARAMETERS:
        ----------
        xr_single_band_stack : xarray.DataArray
            single-band stack of image dates
            of dims:  ('slice', 'y', 'x')
        dask_client : dask.distributed.Client
        drop_nodata : bool
            Drop time slice pixels that are null
            Default = True
        
        ATTRIBUTES:
        ----------
        ts_regress_xr : xarray.DataArray
            two-band Theil-Sen output per pixel
            1) slope of regression
            2) intercept of regression
        """

        def __init__(self, xr_single_band_stack, dask_client, drop_nodata=True, debug=False):        
            if debug:
                print('*** Debug mode ***')

            # Check the input is correct
            _init_err = False
            if not isinstance(xr_single_band_stack, xr.DataArray):
                _init_err = True
            else:
                if len(xr_single_band_stack.shape) != 3:  # must be a single band timeseries
                    _init_err = True
                    
            if _init_err:
                raise Exception('Must pass a single-band raster timeseries as xarray.DataArray ' \
                                "with dims: ('slice', 'y', 'x')")
            
            # if not debug:  # do not use dask client in debug session
            #     # dask_client = get_client()
            #     if not isinstance(dask_client, dask.distributed.Client):
            #         raise Exception('Dask Client must be instantiated before use')
            
                # self._client = dask_client
            
            self._drop_nodata = drop_nodata

            # # convert to dims: slice, y, x
            # # this assumes input has dims: (<something>, "y", "x")
            img = xr_single_band_stack  # by reference, for convenience
            
            # drop all non-dim variables, rename time index variable
            img = img.drop_vars([e for e in img.coords if not e in img.dims]) \
                        .rename({img.dims[0]: 'slice'})
            print(img)
            # reproportion the time coordinate to reflect fractional years
            # starting from the first date of the cube 
            time_of_cube = img.coords['slice']
            timerange = [time_of_cube[e].astype('datetime64[ns]') for e in [0, -1]] # first and last observation
            years = list(map(lambda e: np.datetime64(e.values, 'Y'), timerange)) # first and last as years
            year_span = (years[1] - years[0] + 1).astype(np.int64)

            # original data are datetime64[ns] or nanoseconds
            ns_diff = (time_of_cube[-1] - time_of_cube[0]).astype(np.int64)
            percent_of_cube = (time_of_cube - time_of_cube[0]).astype(np.int64) / ns_diff.astype(np.int64)
            time_in_partyears = percent_of_cube * year_span
            
            self._xr_img = img.assign_coords(slice=time_in_partyears)
            
            self._nbands = self._xr_img.shape[0]

            def xr_to_df(xr_img):
                # Creates a dataframe with:
                #  index: the pixel number
                #  band: the time interval
                #  value: data value
                
                # keep as local dataframes
                src_ds = xr_img.to_dataset(dim='slice')  # group by timeslice
                src_df = src_ds.to_dataframe().reset_index().reset_index()  # get index, y, x as columns            
                src_norm = src_df.drop(columns=['y', 'x']).melt(id_vars='index', var_name='band')
                src_norm = src_norm.sort_values(by=['index', 'band'])
                
                return src_norm
            
            def map_ts_pel(pixel, drop_nodata=True):
                # function that applies the TS regression
                
                # unpack x time coordinate and y value
                pixel = np.array([[t, v] for t, v in pixel]) # {time, value} -> {x, y} 
                
                # set up return type for invalid or null regression
                noval = np.array([np.nan, np.nan]).astype(float).reshape(2, 1, 1)  # output size of TS regression with 1x1 pixels
                
                try:
                    peldf = pd.DataFrame(pixel)
                    if drop_nodata:
                        peldf.dropna(inplace=True)
                    if len(peldf.dropna()) == 0:  # they were all nans
                        return noval

                    # TS regressor requires X in this columnar shape
                    X = peldf[0].values[:, np.newaxis]
                    y = peldf[1].values
                    sp, ns = [200, 10] # see comment below about parameter tuning results

                    # dask parallel backend didn't seem to help as per
                    # https://git.hatfieldgroup.net/Hatfield_Geomatics/aep11232/-/blob/70b62a7328a4d5d596f5fd0973288f6d81b7c458/trend/TS_trend.ipynb
                    # with joblib.parallel_backend('dask'):
                    #     ts_r = TheilSenRegressor(max_subpopulation=sp, n_subsamples=ns, random_state=0, n_jobs=-1, verbose=True).fit(X, y)
                    ts_r = TheilSenRegressor(max_subpopulation=sp, n_subsamples=ns, random_state=0, n_jobs=-1, verbose=True).fit(X, y)

                    slope, intercept = ts_r.coef_[0], ts_r.intercept_

                    Y = np.array([slope, intercept]).reshape(2, 1, 1)
                except:
                    Y = noval
                
                return Y
            
            def map_ts_block(block, timecoords, timelen, drop_nodata=True):
                # This function is called by dask.array.map_blocks over each image chunk
                
                errvals = block * np.nan  # fill with nan
                
                ## map_blocks will occasionally send nonsense like a slice of size (0, 0, 0). What gives?
                try:
                    if len(timecoords) != timelen: # funky map_blocks being map_blocks
                        raise
                    
                    # convert to dataframe
                    block_xr = xr.DataArray(block, dims=['slice', 'y', 'x']) 
                    img_df = xr_to_df(block_xr)
                    
                    # append time coordinate
                    time_df = timecoords.to_pandas().reset_index()['slice'].to_frame()
                    time_df['band'] = time_df.index
                    
                    
                    img_df = pd.merge(left=img_df, right=time_df, how='left', on='band')                
                    img_df = pd.DataFrame( [[i, [t, v]] for i, t, v in img_df[['index', 'slice', 'value']].values.tolist()] , columns=['index', 'time_value'] )

                    # group by index and send each group as an array
                    #  ** there is probably a numpy equivalent like apply_along_axis that might make more sense ** 
                    ts_results = img_df[['index', 'time_value']].groupby('index')['time_value'].apply(lambda e: map_ts_pel(e.values, drop_nodata))

                    # convert results to array with two bands and original block x, y dimensions
                    result = np.concatenate(ts_results.values)
                    result = result.reshape((int(len(result) / 2), 2)).T
                    result = result.reshape((2,) + block.shape[1:])
                    
                    return result
                except:
                    return errvals
            
            def result_to_xr(result):
                # apply the new output and original coords and dims
                out_arr = xr.DataArray(data=result,
                                    coords={'TS_variable': ['slope', 'intercept'], 
                                            'y': self._xr_img.coords['y'],
                                            'x': self._xr_img.coords['x']},
                                    dims=['TS_variable', 'y', 'x'])
                
                # define the CRS if the input image did
                if not self._xr_img.rio.crs is None:
                    out_arr.rio.set_crs(self._xr_img.rio.crs)
                
                return out_arr
            
            # chunk the source image for distributed processing
            self._xr_img_chunk = self._xr_img.chunk((-1, 100, 100))
            
            # calculate TS for each pixel in the block using dask.array.map_blocks
            self._np_result = self._xr_img_chunk.data.map_blocks(map_ts_block, self._xr_img_chunk.coords['slice'], self._xr_img_chunk.shape[0])
            self._np_result.compute()
            # save final result as xarray        
            # self.TS_result = result_to_xr(self._np_result)


def theilsen_trend(pixel):
    slope, intercept, low_slope, high_slope = stats.mstats.theilslopes(pixel)
    return  pd.DataFrame({
         "slope": [slope],
         "intercept": [intercept]})

def trend_block_mapper(array):
    # At this point, things are loaded into memory... do Pandas/numpy
    # print(array.shape)
    df = pd.DataFrame(np.column_stack(list(map(np.ravel, np.meshgrid(*map(np.arange, array.shape), indexing="ij"))) + [array.ravel()]), columns=["band","time", "y", "x", "val"])
    slopes = df.groupby(["x","y","band"], group_keys=True)['val'].apply(theilsen_trend).reset_index()
    indices = slopes.index.tolist()
    slope = slopes["slope"].values.tolist()
    intercepts = slopes["intercept"].values.tolist()
    # print(indices, slope, intercepts)
    ts_slope = np.ones((array.shape[0], array.shape[2], array.shape[3])).flatten()
    ts_int = np.ones((array.shape[0], array.shape[2], array.shape[3])).flatten()
    ts_slope[indices] = slope
    ts_int[indices] = intercepts
    ts_slope = ts_slope.reshape(array.shape[0], array.shape[2], array.shape[3])
    ts_int = ts_int.reshape(array.shape[0], array.shape[2], array.shape[3])
    filler = np.ones((array.shape[0], array.shape[2], array.shape[3]))
    filled = np.repeat([filler], array.shape[1]-2).reshape(array.shape[0], array.shape[1]-2, array.shape[2], array.shape[3])

    ts = np.stack((ts_slope,ts_int), axis=1)
    ts = np.concatenate((ts, filled), axis=1)

    return ts

def _get_slope(array):
    return array[:,0,:,:]

def _get_intercept(intercept):
    return array[:,0,:,:]

def simpler_run(array):
    # apply the function over allpoints to calculate the trend at each point
    array = array.chunk((1, -1, 10, 10))
    array.data = array.data.map_blocks(trend_block_mapper, dtype=np.ndarray)
    # print(type(ts_data))
    # array.data = array[:,0,:,:].squeeze().expand_dims(dim={"parameter":2}, axis=1)
    array = array[:,0:2,:,:].rename({"time":"parameter"})
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

    """
    reco_80 = baseline * (percent / 100)
    ts = simpler_run(image_stack)
    y2r = (reco_80 - ts.sel(parameter="intercept")) / ts.sel(parameter="slope")
    # TODO: maybe return NaN if intercept + slope*curr_year is not recovered
    return y2r

if __name__ == '__main__':


    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster()  # Launches a scheduler and workers locally
    client = Client(cluster)

    import dask.array as da
    import pandas as pd
    import rioxarray


    array = xr.DataArray( da.from_array(np.arange(0, 20).reshape(1, 5, 2, 2), chunks="auto"),
                         coords=[("band", ["red"]),("time",[0]), ("y", np.arange(0,2)), ("x", np.arange(0,2))])
    # TODO (Friday) the baseline dimensions don't match what is happening in spectral_recovery --- fix this then see if the output from simple_years_to_recovery has appropriate dimensions
    # Something about the time dimensions
    bline = xr.DataArray( da.from_array(np.ones(shape=(1, 1, 1,1))*20, chunks="auto"), coords=[("band", ["red"]),("time",[0]), ("y", [0]), ("x", [0])])
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
    print(array.data.compute())
    start3 = timeit.default_timer()
    res = simple_years_to_recovery(array, bline)
    print(res)
    stop3 = timeit.default_timer()
    print(f"simple's: {stop3-start3}")