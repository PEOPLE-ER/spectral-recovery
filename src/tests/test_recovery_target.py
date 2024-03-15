import pytest

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

from shapely import Polygon

from unittest.mock import patch, MagicMock
from xarray.testing import assert_equal
from numpy.testing import assert_array_equal

from spectral_recovery.targets import MedianTarget, WindowedTarget, _buffered_clip_reference_stack, BufferError, compute_recovery_targets


def test_invalid_scale_throws_value_error():
    with pytest.raises(ValueError):
        MedianTarget(scale="not_a_scale")

class TestBufferedClip:

    valid_poly = Polygon([(3.5, 3.5), (3.5, 5.5), (5.5, 5.5), (5.5, 3.5)])

    @pytest.fixture()
    def valid_array(self):
        data = np.ones((1, 5, 10, 10))
        latitudes = np.arange(0, 10)
        longitudes = np.arange(0, 10)
        time = pd.date_range("2010", "2014", freq="YS")
        xarr = xr.DataArray(
            data,
            dims=["band", "time", "y", "x"],
            coords={"band": ["NBR"], "time": time, "y": latitudes[::-1], "x": longitudes},
        )
        xarr = xarr.rio.write_crs("EPSG:3348", inplace=True)
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
            crs="EPSG:3348",
        )
        return valid_frame

    def test_returns_buffered_clip_sliced_to_reference_years(self, valid_array, valid_frame):
        buffer = 1
        square_side = 2
        time = pd.date_range("2012", "2013", freq="YS")
        expected_result = xr.DataArray(
            np.ones((1, 2, buffer*2+square_side, buffer*2+square_side)),
            dims=["band", "time", "y", "x"],
            coords={"band": ["NBR"], "time": time, "y": [3, 4, 5, 6], "x": [3, 4, 5, 6]},
        )
        print(valid_array)
        result = _buffered_clip_reference_stack(valid_array, valid_frame, "2012", "2013", buffer=buffer)
        result = result.drop_vars("spatial_ref")

        assert result.equals(expected_result)
    
    # def test_buffer_beyond_bounds_returns_value_err_with_pad_values(self, valid_array, valid_frame):
    #     buffer = 5
    #     expected_y_back = 2
    #     expected_y_front = 2
    #     expected_x_back = 2
    #     expected_x_front = 2

    #     with pytest.raises(BufferError) as b_info:
    #         _buffered_clip_reference_stack(valid_array, valid_frame, "2012", "2013", buffer=buffer)
    
    #     assert b_info.value.y_back == expected_y_back
    #     assert b_info.value.y_front == expected_y_front
    #     assert b_info.value.x_back == expected_x_back
    #     assert b_info.value.x_front == expected_x_front

    # def test_buffer_beyond_one_bound_returns_value_err_with_one_pad_value(self, valid_array, valid_frame):
    #     buffer = 2
    #     expected_y_back = 1
    #     expected_y_front = 1
    #     expected_x_back = 2
    #     expected_x_front = 1

    #     valid_frame.at[0, 'geometry'] = Polygon([(0.5, 3.5), (0.5, 5.5), (5.5, 5.5), (5.5, 3.5), (0.5, 3.5)])

    #     with pytest.raises(BufferError) as b_info:
    #         _buffered_clip_reference_stack(valid_array, valid_frame, "2012", "2013", buffer=buffer)
    
    #     assert b_info.value.y_back == expected_y_back
    #     assert b_info.value.y_front == expected_y_front
    #     assert b_info.value.x_back == expected_x_back
    #     assert b_info.value.x_front == expected_x_front

# class TestComputeRecoveryTargetsMedian:

# def test_median_recovery_targets_is_return(self, valid_array, valid_frame):
#     # Mock objects
#     median_target_mock = MagicMock(spec=MedianTarget)

#     # Mock behaviors
#     median_target_mock.return_value = "Mocked median target"

#     # Test with MedianTarget
#     result_median = compute_recovery_targets(
#         timeseries=timeseries,
#         restoration_polygon=restoration_polygon,
#         reference_start="2012",
#         reference_end="2013",
#         func=median_target_mock
#     )
#     assert result_median == "Mocked median target"

class TestComputeRecoveryTargetsWindowed:

    valid_poly = Polygon([(3.5, 3.5), (3.5, 5.5), (5.5, 5.5), (5.5, 3.5)])

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
        xarr = xarr.rio.write_crs("EPSG:3348", inplace=True)
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
            crs="EPSG:3348",
        )
        return valid_frame
    
    @patch("rioxarray.raster_array.RasterArray.clip")
    @patch("spectral_recovery.targets._buffered_clip_reference_stack")
    def test_windowed_recovery_targets_is_clipped_return(self, buffer_clip_mock, clip_mock, valid_array, valid_frame):
        windowed_target_mock = MagicMock(spec=WindowedTarget)
        windowed_target_mock.N = 3
        windowed_target_mock.return_value = valid_array
        clip_mock.return_value = "Mocked clip return"

        result_windowed = compute_recovery_targets(
            timeseries=valid_array,
            restoration_polygon=valid_frame,
            reference_start="2010",
            reference_end="2010",
            func=windowed_target_mock
        )
        assert result_windowed == "Mocked clip return"

    @patch("spectral_recovery.targets._buffered_clip_reference_stack")
    def test_calls_buffer_clip_once(self, buffer_clip_mock, valid_array, valid_frame):
        buffer_clip_mock.side_effect = valid_array
        windowed_method = WindowedTarget(N=13)

        compute_recovery_targets(
            timeseries=valid_array,
            restoration_polygon=valid_frame,
            reference_start="2010",
            reference_end="2010",
            func=windowed_method
        )

        assert buffer_clip_mock.call_count == 1
        _, _, call_kwargs = buffer_clip_mock.mock_calls[0]
        assert_equal(call_kwargs["timeseries"], valid_array)
        pd.testing.assert_frame_equal(call_kwargs["restoration_polygon"], valid_frame)
        assert call_kwargs["reference_start"] == "2010"
        assert call_kwargs["reference_end"] == "2010"
        assert call_kwargs["buffer"] == (windowed_method.N-1)/2


    # @patch("spectral_recovery.targets._buffered_clip_reference_stack")
    # @patch("xarray.DataArray.pad")
    # def test_buffer_error_and_pad_true_recalls_clip_with_padded_array(self, pad_mock, buffer_clip_mock, valid_array, valid_frame):
    #     buffer_clip_mock.side_effect = [BufferError("msg", 1, 2, 3, 4), valid_array]
    #     pad_mock.return_value = valid_array * 2
    #     windowed_method = WindowedTarget(N=13, pad=True)

    #     compute_recovery_targets(
    #         timeseries=valid_array,
    #         restoration_polygon=valid_frame,
    #         reference_start="2010",
    #         reference_end="2010",
    #         func=windowed_method
    #     )

    #     assert buffer_clip_mock.call_count == 2
    #     # Test that first call usses the passed arguments
    #     pad_mock.assert_called_with(x=(3, 4), y=(1, 2), mode="edge")
    #     _, _, call_kwrgs_1 = buffer_clip_mock.mock_calls[0]
    #     assert_equal(call_kwrgs_1["timeseries"], valid_array)
    #     pd.testing.assert_frame_equal(call_kwrgs_1["restoration_polygon"], valid_frame) 
    #     assert call_kwrgs_1["reference_start"] == "2010"
    #     assert call_kwrgs_1["reference_end"] == "2010"
    #     assert call_kwrgs_1["buffer"] == (windowed_method.N-1)/2
    #     # Test that second call uses padded arrays + the same other args
    #     _, _, call_kwrgs_2 = buffer_clip_mock.mock_calls[1]
    #     assert_equal(call_kwrgs_2["timeseries"], pad_mock.return_value)
    #     pd.testing.assert_frame_equal(call_kwrgs_2["restoration_polygon"], valid_frame)
    #     assert call_kwrgs_2["reference_start"] == "2010"
    #     assert call_kwrgs_2["reference_end"] == "2010"
    #     assert call_kwrgs_2["buffer"] == (windowed_method.N-1)/2
    
    # def test_buffer_error_and_pad_false_raises_buff_err(self, valid_array, valid_frame):
    #     windowed_method = WindowedTarget(N=13, pad=False)

    #     with pytest.raises(BufferError):
    #         compute_recovery_targets(
    #             timeseries=valid_array,
    #             restoration_polygon=valid_frame,
    #             reference_start="2010",
    #             reference_end="2010",
    #             func=windowed_method
    #         )
    

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
        out_stack = median_polygon_method(test_stack)

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
        out_stack = median_polygon_method(test_stack)

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
        out_stack = median_polygon_method(test_stack)

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
        out_stack = median_polygon_method(test_stack)

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
        out_stack = median_polygon_method(test_stack)

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
        out_stack = median_pixel_method(test_stack)

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
        out_stack = median_pixel_method(test_stack)

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
            np.zeros((4, 1, 2, 2)),
            dims=input_dims,
            coords={"time": [1]} 
        )
        expected_dims_and_sizes = {"band": 4, "y": 2, "x": 2}
        windowed_method = WindowedTarget()

        result = windowed_method(input_data)

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

        result_data = windowed_method(input_data).data

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

        result_data = windowed_method(input_data).data

        assert_array_equal(expected_data, result_data)
    
    def test_nan_rm_true_computes_without_NaN(self):
        """
        Given:

            o, o, o, o
            o, x, x, o
            o, x, o, o
            o, o, o, o
        
        where o is NaN space and x is non-Nan/polygon space, ensure
        WindowedTarget returns values for all, and only, x locations.
        
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
                [1.0, 1.5, 1.5, 2.0],
                [3.5, 3.0, 3.0, 2.0],
                [3.5, 3.0, 3.0, 2.0],
                [6.0, 6.0, 6.0, np.nan],
            ]]
        )
        windowed_method = WindowedTarget(N=3, na_rm=True)

        result_data = windowed_method(input_data).data

        assert_array_equal(expected_data, result_data)
    
    def test_nan_rm_false_computes_with_NaN(self):
        """
        Given an array like:

            o, o, o, o
            x, x, x, o
            x, !, x, o
            x, x, x, o
        
        where o is NaN space and x/! is non-Nan space,
        ensure WindowedTarget return only values for values
        with a full 3x3 window (i.e !).
        
        """
        data = [[[
            [np.nan, np.nan, np.nan, np.nan],
            [     1,      1,   5, np.nan],
            [     1,      6,   1, np.nan],
            [     1,      1,   1, np.nan],
        ]]]
        input_data = xr.DataArray(
            data,
            dims=["band", "time", "y", "x"],
            coords={"time": [1]}
        )
        expected_data = np.array([[
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, 2.0,    np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
            ]]
        )
        windowed_method = WindowedTarget(N=3, na_rm=False)

        result_data = windowed_method(input_data).data

        assert_array_equal(expected_data, result_data)
    



