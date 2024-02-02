import pytest 

import numpy as np
import xarray as xr


from xarray.testing import assert_equal
from spectral_recovery.recovery_target import make_median_target


def test_invalid_scale_throws_value_error():
    with pytest.raises(ValueError):
        make_median_target(scale="not_a_scale")


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

        median_polygon_method = make_median_target(scale="polygon")
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
        median_polygon_method = make_median_target(scale="polygon")
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
        median_polygon_method = make_median_target(scale="polygon")
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
        median_polygon_method = make_median_target(scale="polygon")
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
        median_polygon_method = make_median_target(scale="polygon")
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
        median_pixel_method = make_median_target(scale="pixel")
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
        median_pixel_method = make_median_target(scale="pixel")
        out_stack = median_pixel_method(test_stack, [0, 1])

        assert_equal(out_stack, expected_stack)
