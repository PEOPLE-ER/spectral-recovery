import pytest
import rioxarray

import numpy as np
import geopandas as gpd
import xarray as xr

from shapely import Polygon

from xarray.testing import assert_equal
from spectral_recovery.baselines import historic_average



class TestHistoricAverage():

    def test_no_nan_returns_avg_over_time(self):
        test_data = [
            [
                [[1.0]], # Time 1, band 1
                [[1.0]], # Time 1, band 2
            ],
            [
                [[3.0]], # Time 2, band 1
                [[5.0]], # Time 2, band 2
            ]

        ]
        test_stack = xr.DataArray(
            test_data,
            dims=["time", "band", "y", "x"],
            coords={
                "time": [0, 1],
            },
        )
        expected_data = [2.0, 3.0]
        expected_stack = xr.DataArray(
            expected_data,
            dims=["band"],
            coords={
                "band": [0,1],
            },
        )
        out_stack = historic_average(test_stack, (0, 1))
        assert_equal(out_stack, expected_stack)
    
    def test_nan_timeseries_is_nan(self):
        test_data = [
            [
                [[np.nan]],
                [[3.0]],
            ],
            [
                [[np.nan]],
                [[5.0]],
            ]

        ]
        test_stack = xr.DataArray(
            test_data,
            dims=["time", "band", "y", "x"],
            coords={
                "time": [0, 1],
            },
        )
        expected_data = [np.nan, 4.0]
        expected_stack = xr.DataArray(
            expected_data,
            dims=["band"],
            coords={
                "band": [0, 1],
            },
        )
        out_stack = historic_average(test_stack, (0, 1))
        assert_equal(out_stack, expected_stack)
    
    def test_nan_in_timeseries_ignored(self):
        test_data = [
            [
                [[np.nan]],
                [[3.0]],
            ],
            [
                [[9.0]],
                [[5.0]],
            ]

        ]
        test_stack = xr.DataArray(
            test_data,
            dims=["time", "band", "y", "x"],
            coords={
                "time": [0, 1],
            },
        )
        expected_data = [9.0, 4.0]
        expected_stack = xr.DataArray(
            expected_data,
            dims=["band"],
            coords={
                "band": [0, 1],
            },
        )
        out_stack = historic_average(test_stack, (0, 1))
        assert_equal(out_stack, expected_stack)
    
    def test_multi_poly_returns_correct_avg(self):
        test_data = [
            [
                [
                    [
                        [1.0, 1.0], # y 1, x1
                        [np.nan, np.nan] # y 2, x1
                    ], # band 1
                ],
                [
                    [
                        [3.0, 6.0], # y 1
                        [np.nan, np.nan]
                    ], # band 1
                ]

            ],
            [
                
                [
                    [
                        [np.nan, np.nan], # y 1, x1
                        [np.nan, 2.0] # y 2, x1
                    ], # band 1
                ],
                [
                    [
                        [np.nan, np.nan], # y 1
                        [np.nan, 6.0]
                    ], # band 1
                ]

            ]
        ]
        test_stack = xr.DataArray(
            test_data,
            dims=["poly_id", "time", "band", "y", "x"],
            coords={
                "time": [0, 1],
            },
        )
        expected_data = [3.375] 
        expected_stack = xr.DataArray(
            expected_data,
            dims=["band"],
            coords={
                "band": [0],
            },
        )
        out_stack = historic_average(test_stack, (0, 1))
        assert_equal(out_stack, expected_stack)

