import pytest

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

from shapely import Polygon
from xarray.testing import assert_equal
from numpy.testing import assert_array_equal


from spectral_recovery.targets import MedianTarget, WindowedTarget, _buffered_clip_reference_stack


def test_invalid_scale_throws_value_error():
    with pytest.raises(ValueError):
        MedianTarget(scale="not_a_scale")

class TestBufferedClip:

    valid_poly = Polygon([(4, 4), (4, 5), (5, 5), (5, 4), (4, 4)])

    @pytest.fixture()
    def valid_array(self):
        data = np.ones((1, 5, 10, 10))
        latitudes = np.arange(0, 10)
        longitudes = np.arange(0, 10)
        time = pd.date_range("2010", "2014", freq="YS")
        xarr = xr.DataArray(
            data,
            dims=["band", "time", "y", "x"],
            coords={"band": ["NBR"], "time": time, "y": latitudes, "x": longitudes},
        )
        xarr.rio.write_crs("EPSG:4326", inplace=True)
        return xarr

    @pytest.fixture()
    def valid_frame(self):

        valid_frame = gpd.GeoDataFrame(
            {
                "dist_start": [2012],
                "rest_start": [2013],
                "reference_start": [2010],
                "reference_end": [2010],
                "geometry": [self.valid_poly],
            },
            crs="EPSG:4326",
        )
        return valid_frame

    def test_buffered_clip_returns_correct_clip(self, valid_array, valid_frame):
        result = _buffered_clip_reference_stack(valid_array, valid_frame, "2012", "2013", buffer=1)
        print(result)


class TestComputeRecoveryTargets:

    def test_reference_with_median_pixel_method_throws_value_err():
        raise NotImplementedError
    
    def test_reference_with_windowed_mean_throws_value_err():
        raise NotImplementedError
    
    def test_median_target_called_with_correct_reference_stack():
        raise NotImplementedError
    
    def test_windowed_target_called_with_correct_reference_stack():
        raise NotImplementedError
    

class TestMedianTargetPolygonScale:
    def test_no_nan_returns_avg_over_time(self):
        test_data = [
            [
                [[1.0]],  # Time 1, band 1
                [[1.0]],  # Time 1, band 2
            ],
            [
                [[3.0]],  # Time 2, band 1
                [[5.0]],  # Time 2, band 2
            ],
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
                "band": [0, 1],
            },
        )

        median_polygon_method = MedianTarget(scale="polygon")
        out_stack = median_polygon_method(test_stack, [0, 1])

        assert_equal(out_stack, expected_stack)

    def test_odd_time_dim_returns_median(self):
        test_data = [
            [
                [[1.0]],  # Time 1, band 1
                [[1.0]],  # Time 1, band 2
            ],
            [
                [[3.0]],  # Time 2, band 1
                [[5.0]],  # Time 2, band 2
            ],
            [
                [[9.0]],  # Time 3, band 1
                [[7.0]],  # Time 3, band 2
            ],
        ]
        test_stack = xr.DataArray(
            test_data,
            dims=["time", "band", "y", "x"],
            coords={
                "time": [0, 1, 2],
            },
        )
        expected_data = [3.0, 5.0]
        expected_stack = xr.DataArray(
            expected_data,
            dims=["band"],
            coords={
                "band": [0, 1],
            },
        )
        median_polygon_method = MedianTarget(scale="polygon")
        out_stack = median_polygon_method(test_stack, [0, 2])

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
            ],
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
        median_polygon_method = MedianTarget(scale="polygon")
        out_stack = median_polygon_method(test_stack, [0, 1])

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
            ],
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
        median_polygon_method = MedianTarget(scale="polygon")
        out_stack = median_polygon_method(test_stack, [0, 1])

        assert_equal(out_stack, expected_stack)

    def test_multi_poly_averages_individual_polygon(self):
        test_data = [
            [  # Polygon 1
                [  # Time 1
                    [[1.0, 2.0], [3.0, 4.0]],  # y1, x1  # y2, x1  # band 1
                ],
                [  # Time 2
                    [[5.0, 6.0], [8.0, 9.0]],  # y1, x2   # band 1
                ],
            ],
            [  # Polygon 2
                [
                    [[1.0, 2.0], [3.0, 4.0]],  # y 1, x1  # y 2, x1  # band 1
                ],
                [
                    [[5.0, 6.0], [8.0, 9.0]],  # y 1  # band 1
                ],
            ],
        ]
        test_stack = xr.DataArray(
            test_data,
            dims=["poly_id", "time", "band", "y", "x"],
            coords={
                "time": [0, 1],
            },
        )
        expected_data = [4.75]
        expected_stack = xr.DataArray(
            expected_data,
            dims=["band"],
            coords={
                "band": [0],
            },
        )
        median_polygon_method = MedianTarget(scale="polygon")
        out_stack = median_polygon_method(test_stack, [0, 1])

        assert_equal(out_stack, expected_stack)


class TestMedianTargetPixelScale:
    def test_scale_pixel_returns_correct_dimensions(self):
        test_data = np.arange(8).reshape(1, 2, 2, 2)
        test_stack = xr.DataArray(
            test_data,
            dims=["band", "time", "y", "x"],
            coords={
                "time": [0, 1],
            },
        )
        median_pixel_method = MedianTarget(scale="pixel")
        out_stack = median_pixel_method(test_stack, [0, 1])

        assert out_stack.dims == ("band", "y", "x")
        assert out_stack.shape == (1, 2, 2)

    def test_scale_pixel_returns_per_pixel_median(self):
        test_data = [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [8.0, 9.0]],
            ],
        ]
        test_stack = xr.DataArray(
            test_data,
            dims=["band", "time", "y", "x"],
            coords={
                "time": [0, 1],
            },
        )

        expected_data = [[[3.0, 4.0], [5.5, 6.5]]]
        expected_stack = xr.DataArray(
            expected_data,
            dims=["band", "y", "x"],
            coords={
                "band": [0],
            },
        )
        median_pixel_method = MedianTarget(scale="pixel")
        out_stack = median_pixel_method(test_stack, [0, 1])

        assert_equal(out_stack, expected_stack)


class TestWindowedTarget:

    def test_neg_or_0_N_throws_value_err(self):
        with pytest.raises(
            ValueError, 
        ):
            WindowedTarget(N=-1)

    def test_even_N_throws_value_err(self):
        with pytest.raises(
            ValueError, 
        ):
            WindowedTarget(N=2)

    def test_default_window_size_is_3(self):
        windowed_method = WindowedTarget()
        assert windowed_method.N == 3

    def test_window_returns_correct_dims(self):
        input_dims = ["band", "time", "y", "x"]
        input_data = xr.DataArray(
            np.zeros((4, 3, 2, 2)),
            dims=input_dims,
            coords={"time": [1, 2, 3]} 
        )
        expected_dims_and_sizes = {"band": 4, "y": 2, "x": 2}
        windowed_method = WindowedTarget()

        result = windowed_method(input_data, reference_date=1)

        assert len(result.dims) == len(expected_dims_and_sizes.keys())
        for dim in result.dims:
            assert result.sizes[dim] == expected_dims_and_sizes[dim]

    def test_default_window_returns_correct_target_values(self):
        data = np.arange(-8, 19).reshape((1, 3, 3, 3))
        input_data = xr.DataArray(
            data,
            dims=["band", "time", "y", "x"],
            coords={"time": [1, 2, 3]}
        )
        expected_data = np.array([[[np.nan, np.nan, np.nan],[np.nan, 5.0, np.nan], [np.nan, np.nan, np.nan]]])
        windowed_method = WindowedTarget()

        result_data = windowed_method(input_data, reference_date=[1, 3]).data

        assert_array_equal(expected_data, result_data)

    def test_default_window_returns_correct_target_values(self):
        data = np.arange(-8, 19).reshape((1, 3, 3, 3))
        input_data = xr.DataArray(
            data,
            dims=["band", "time", "y", "x"],
            coords={"time": [1, 2, 3]}
        )
        # Median of `data` along time dim will be 3x3 array with values 1-9. Mean of 5.
        # Since only the centre pixel can have a full 3x3 window, centre should be set 
        # to 5 and all others should be NaN.
        expected_data = np.array([[[np.nan, np.nan, np.nan],[np.nan, 5.0, np.nan], [np.nan, np.nan, np.nan]]])
        windowed_method = WindowedTarget()

        result_data = windowed_method(input_data, reference_date=[1, 3]).data

        assert_array_equal(expected_data, result_data)
    
    def test_polygon_borders_maintained(self):
        """
        Represent a polygon in an array like:

            o, o, o, o
            o, x, x, o
            o, x, o, o
            o, o, o, o
        
        where o is NaN/non-polygon space and x is non-Nan/polygon space.
        Ensure WindowedTarget returns values for all, and only, x locations.
        
        """
        data = [[[
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan,      1,      2, np.nan],
            [np.nan,      6, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ]]]
        input_data = xr.DataArray(
            data,
            dims=["band", "time", "y", "x"],
            coords={"time": [1]}
        )
        expected_data = np.array([[
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan,    3.0,    3.0, np.nan],
                [np.nan,    3.0, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ]]
        )
        windowed_method = WindowedTarget()

        result_data = windowed_method(input_data, reference_date=[1, 1]).data

        assert_array_equal(expected_data, result_data)
    



