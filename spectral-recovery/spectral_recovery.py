import rioxarray
import dask
import os
os.environ['USE_PYGEOS'] = '0'
import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, Union, List, Dict

from restoration import RestorationArea
from images import MultiBandYearlyStack
from metrics import percent_recovered, Metrics



def spectral_recovery(
        band_dict: Dict[str, xr.DataArray],
        restoration_poly: gpd.GeoDataFrame,
        restoration_year: int,
        reference_years: Union[int, List[int]],
        indices_list: List[str],
        metrics_list: List[str],
        data_mask: xr.DataArray = None
        ) -> None:
        """
        The main calling function. Better doc-string incoming. 

        Parameters
        -----------
        restoration_poly : str or Polygon
            Path to vector file or Polygon object with polygon
            representing restoration area

        restoration_year :
            Year of restoration event.

        """
        indices = MultiBandYearlyStack(
             bands=band_dict,
             dict=True,
             data_mask=data_mask
             ).indices(indices_list) # NOTE: indices computed on full stack. Might come to regret this.
        ra = RestorationArea(
             restoration_polygon=restoration_poly,
             restoration_year=restoration_year,
             reference_system=reference_years,
             composite_stack=indices
        )
        metrics = ra.metrics(metrics_list)
        a = metrics.sel(metric='years_to_recovery').data.compute()
        print(a)
        # data = metrics.sel(metric="years_to_recovery").data.compute()[0][0][0]
        # vals = ra.stack.sel(time=slice(ra.restoration_year,ra.end_year)).data.compute()
        # intercept =  ra.stack.sel(time=slice(ra.restoration_year)).squeeze().data.compute()
        # y_vals = intercept[0] + data[0]*vals
        # y_vals = y_vals[0].flatten()
        # x_vals = ra.stack["time"].dt.year.compute()[5:]

        # print(y_vals, x_vals)
        # plt.plot(x_vals, y_vals, '--')
        # plt.plot(x_vals, vals[0].flatten(), 'o-')
        # plt.show()

        return


if __name__ == "__main__":
    test_poly = gpd.read_file("../../data/smaller_poly.gpkg")
    bad_poly = gpd.read_file("../../data/out_of_bounds_poly.gpkg")
    rest_year = pd.to_datetime("2013")
    reference_year = (pd.to_datetime("2010"), pd.to_datetime("2012"))

    test_stack = rioxarray.open_rasterio("../../data/nir_18_19.tif",
                                             chunks="auto")
    test_stack = xr.ones_like(test_stack)
    test_stack = test_stack.rename({"band":"time"})
    
    test_stack =  xr.concat(
            [test_stack*100, test_stack*100,test_stack*101, test_stack*100, test_stack*11, test_stack,test_stack*2, test_stack*2.5, test_stack*3, test_stack*3.5, test_stack*4, test_stack*4.5], 
            dim=pd.Index(["2008","2009","2010", "2011","2012","2013","2014", "2015","2016","2017","2018", "2019"], name="time")
            )
    # test_mask = xr.where(test_stack > 15515, True, False)
    test_stack = test_stack.assign_coords(time=(pd.to_datetime(["2008","2009","2010", "2011","2012","2013","2014", "2015","2016","2017","2018", "2019"])))
    # test_stack.rio.write_crs(test_raster.rio.crs, inplace=True)
    # test_stack.rio.update_encoding(test_stack.encoding, inplace=True)

    # print(test_stack.rio.crs, test_stack.encoding, test_stack.attrs)

    test_band_dict = {"red": test_stack * np.random.randint(low=1, high=50, size=test_stack.shape), 
                      "nir": test_stack * np.random.randint(low=1, high=50, size=test_stack.shape),
                      "swir": test_stack * np.random.randint(low=1, high=50, size=test_stack.shape)
                      }
    
    spectral_recovery(
         band_dict=test_band_dict,
         restoration_poly=test_poly,
         restoration_year=rest_year,
         reference_years=reference_year,
         indices_list=["ndvi"],
         metrics_list=["years_to_recovery"],
         )