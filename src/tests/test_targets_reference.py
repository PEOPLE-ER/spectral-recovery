import pytest

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

from shapely import Polygon
from xarray.testing import assert_equal
from unittest.mock import patch

from spectral_recovery.targets.reference import median, _check_reference_years

class TestMedian:

    @pytest.fixture()
    def valid_gpd(self):
        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        valid_gpd = gpd.GeoDataFrame(geometry=[polygon]).set_crs("EPSG:4326")
        return valid_gpd

    @patch("geopandas.read_file")
    def test_str_polygon_reads_polygon(self, read_mock, valid_gpd):
        read_mock.return_value = valid_gpd
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
        valid_stack = xr.DataArray(
            test_data,
            dims=["time", "band", "y", "x"],
            coords={
                "time": pd.date_range("2010", "2011", freq="YS"),
                "y": [1],
                "x": [1]
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        _ = median(reference_sites="pathy", timeseries_data=valid_stack, reference_start=2010, reference_end=2011)

        read_mock.assert_called_once()
        assert read_mock.call_args.args[0] == "pathy"
    
    def test_no_nan_returns_avg_over_time(self, valid_gpd):
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
                "time": pd.date_range("2010", "2011", freq="YS"),
                "y": [1],
                "x": [1]
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        expected_data = [2.0, 3.0]
        expected_stack = xr.DataArray(
            expected_data,
            dims=["band"],
            coords={
                "band": [0, 1]
            },
        ).rio.write_crs("EPSG:4326", inplace=True)

        out_targets = median(reference_sites=valid_gpd, timeseries_data=test_stack, reference_start=2010, reference_end=2011)

        assert_equal(expected_stack, out_targets)

    def test_odd_time_dim_returns_median(self, valid_gpd):
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
                "time": pd.date_range("2010", "2012", freq="YS"),
                "y": [1],
                "x": [1]
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        expected_data = [3.0, 5.0]
        expected_stack = xr.DataArray(
            expected_data,
            dims=["band"],
            coords={
                "band": [0, 1],
            },
        ).rio.write_crs("EPSG:4326", inplace=True)

        out_stack = median(reference_sites=valid_gpd, timeseries_data=test_stack, reference_start=2010, reference_end=2012)
        assert_equal(expected_stack, out_stack)


    def test_nan_timeseries_is_nan(self, valid_gpd):
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
                "time": pd.date_range("2010", "2011", freq="YS"),
                "y": [1],
                "x": [1]
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        expected_data = [np.nan, 4.0]
        expected_stack = xr.DataArray(
            expected_data,
            dims=["band"],
            coords={
                "band": [0, 1]
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        out_stack = median(reference_sites=valid_gpd, timeseries_data=test_stack, reference_start=2010, reference_end=2011)

        assert_equal(expected_stack, out_stack)

    def test_nan_in_timeseries_ignored(self, valid_gpd):
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
                "time": pd.date_range("2010", "2011", freq="YS"),
                "y": [1],
                "x": [1]
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        expected_data = [9.0, 4.0]
        expected_stack = xr.DataArray(
            expected_data,
            dims=["band"],
            coords={
                "band": [0, 1],
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        out_stack = median(reference_sites=valid_gpd, timeseries_data=test_stack, reference_start=2010, reference_end=2011)
        assert_equal(expected_stack, out_stack)
    
    def test_one_value_per_band_returned(self):
        polygon = Polygon([(-1, 5), (-1, 5), (5, 5), (5, -1)])
        valid_gpd = gpd.GeoDataFrame(geometry=[polygon]).set_crs("EPSG:4326")
        test_data = np.ones((3, 2, 5, 5))
        test_stack = xr.DataArray(
            test_data,
            dims=["band", "time", "y", "x"],
            coords={
                "time": pd.date_range("2010", "2011", freq="YS"),
                "y": np.arange(5),
                "x": np.arange(5)
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        expected_data = [1.0, 1.0, 1.0]
        expected_stack = xr.DataArray(
            expected_data,
            dims=["band"],
            coords={
                "band": [0, 1, 2],
            },
        ).rio.write_crs("EPSG:4326", inplace=True)
        out_stack = median(reference_sites=valid_gpd, timeseries_data=test_stack, reference_start=2010, reference_end=2011)
        
        assert out_stack.dims == ("band",)
        assert len(out_stack.band) == 3
        assert_equal(expected_stack, out_stack)

class TestCheckReferenceYears:

    test_stack = xr.DataArray(
                [[[[1.],[4.],[7.]]]],
                dims=["time", "band", "y", "x"],
                coords={
                    "time": pd.date_range("2010", "2010", freq="YS"),
                    "y": [-1, 0, 1],
                    "x": [0]
                },
        ).rio.write_crs("EPSG:3348", inplace=True)

    def test_valid_reference_years_successful(self):
        _check_reference_years(
            timeseries_data=self.test_stack,
            reference_start=2010,
            reference_end=2010
        )

    def test_invalid_ref_years_throws_value_err(self):
        with pytest.raises(ValueError):
            _check_reference_years(
                timeseries_data=self.test_stack,
                reference_start=10,
                reference_end=20112
            )
    
    def test_oob_start_year_throws_value_err(self):
        with pytest.raises(ValueError):
            _check_reference_years(
                timeseries_data=self.test_stack,
                reference_start=2009,
                reference_end=2010
            )
    
    def test_oob_end_year_throws_value_err(self):
        with pytest.raises(ValueError):
            _check_reference_years(
                timeseries_data=self.test_stack,
                reference_start=2010,
                reference_end=2011
            )

class TestMedianMultipleSites:

     def test_averages_over_reference_sites(self):
            
            polygon1 = Polygon([(-2, 2), (0.75, 2), (0.75, -1.75), (-2, -2), (-2, 2)])
            polygon2 = Polygon([(0.75, 2), (0, 2), (2, 2), (0.75, -1.75), (0.75, 2)])
            valid_gpd = gpd.GeoDataFrame(geometry=[polygon1, polygon2]).set_crs("EPSG:3348")

            test_data = [
                    [
                        [
                            [1., 2., 3.],
                            [4., 5., 6.],
                            [7., 8., 9.]
                        ]
                    ],

                    [[[10., 11., 12.],
                    [13., 14., 15.],
                    [16., 17., 18.]]]
            ]
    
            test_stack = xr.DataArray(
                test_data,
                dims=["time", "band", "y", "x"],
                coords={
                    "time": pd.date_range("2010", "2011", freq="YS"),
                    "y": [1, 0, -1],
                    "x": [-1, 0, 1]
                },
            ).rio.write_crs("EPSG:3348", inplace=True)
            expected_data = [9.75]
            expected_stack = xr.DataArray(
                expected_data,
                dims=["band"],
                coords={
                    "band": [0],
                },
            ).rio.write_crs("EPSG:4326", inplace=True)

            out_stack = median(reference_sites=valid_gpd, timeseries_data=test_stack, reference_start="2010", reference_end="2011")
            assert_equal(out_stack, expected_stack)