import pytest
import rioxarray

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from numpy.testing import assert_almost_equal
from spectral_recovery.config import DATETIME_FREQ
from spectral_recovery.timeseries import _datetime_to_index, _SatelliteTimeSeries


class TestSatelliteTimeSeriesContainsSpatial:
    @pytest.fixture()
    def image_stack(self):
        test_raster = "src/tests/test_data/time3_xy2_epsg3005.tif"
        with rioxarray.open_rasterio(test_raster) as data:
            test_stack = data
            test_stack = test_stack.rename({"band": "time"})
            test_stack = test_stack.expand_dims(dim={"band": [0]})
            test_stack = test_stack.assign_coords(
                time=pd.date_range("2007", "2009", freq=DATETIME_FREQ)
            )
        return test_stack

    def test_inbound_polygon_returns_true(self, image_stack):
        test_poly = gpd.read_file("src/tests/test_data/polygon_inbound_epsg3005.gpkg")
        assert image_stack.satts.contains_spatial(test_poly)

    def test_outbound_polygon_returns_false(self, image_stack):
        test_poly = gpd.read_file("src/tests/test_data/polygon_outbound_epsg3005.gpkg")
        assert not image_stack.satts.contains_spatial(test_poly)

    def test_overlap_polygon_returns_false(self, image_stack):
        test_poly = gpd.read_file("src/tests/test_data/polygon_overlap_epsg3005.gpkg")
        assert not image_stack.satts.contains_spatial(test_poly)

    def test_inbound_multipolygon_returns_true(self, image_stack):
        test_poly = gpd.read_file(
            "src/tests/test_data/polygon_multi_inbound_epsg3005.gpkg"
        )
        assert image_stack.satts.contains_spatial(test_poly)

    def test_overlap_multipolygon_returns_true(self, image_stack):
        test_poly = gpd.read_file(
            "src/tests/test_data/polygon_multi_inoutbound_epsg3005.gpkg"
        )
        assert not image_stack.satts.contains_spatial(test_poly)


class TestSatelliteTimeSeriesContainsTemporal:
    @pytest.fixture()
    def image_stack(self):
        test_raster = "src/tests/test_data/time3_xy2_epsg3005.tif"
        with rioxarray.open_rasterio(test_raster) as data:
            test_stack = data
            test_stack = test_stack.rename({"band": "time"})
            test_stack = test_stack.expand_dims(dim={"band": [0]})
            test_stack = test_stack.assign_coords(
                time=pd.date_range("2007", "2009", freq=DATETIME_FREQ)
            )
        return test_stack

    def test_inbound_date_returns_true(self, image_stack):
        test_date = pd.to_datetime("2008")
        assert image_stack.satts.contains_temporal(test_date)

    def test_outbound_future_date_returns_false(self, image_stack):
        test_date = pd.to_datetime("2010")
        assert not image_stack.satts.contains_temporal(test_date)

    def test_outbound_past_date_returns_false(seld, image_stack):
        test_date = pd.to_datetime("2006")
        assert not image_stack.satts.contains_temporal(test_date)

    def test_inbound_date_range_returns_true(self, image_stack):
        test_date = [pd.to_datetime("2008"), pd.to_datetime("2009")]
        assert image_stack.satts.contains_temporal(test_date)

    def test_outbound_date_range_returns_false(self, image_stack):
        test_date = [pd.to_datetime("2005"), pd.to_datetime("2006")]
        assert not image_stack.satts.contains_temporal(test_date)

    def test_overlap_date_range_returns_false(self, image_stack):
        test_date = [pd.to_datetime("2006"), pd.to_datetime("2008")]
        assert not image_stack.satts.contains_temporal(test_date)


class TestSatelliteTimeSeriesStats:
    @pytest.fixture()
    def test_stack(self):
        test_data = np.arange(16).reshape(2, 2, 2, 2)
        test_stack = xr.DataArray(
            test_data,
            dims=["band", "time", "y", "x"],
            coords={
                "time": pd.date_range("2007", "2008", freq=DATETIME_FREQ),
            },
        )
        return test_stack

    def test_stats_returns_correct_dims(self, test_stack):
        stats = test_stack.satts.stats()
        assert stats.dims == ("stats", "band", "time")

    def test_stats_returns_correct_coords(self, test_stack):
        stats = test_stack.satts.stats()
        assert (stats["stats"].values == ["mean", "median", "max", "min", "std"]).all()

    def test_stats_returns_correct_median(self, test_stack):
        stats = test_stack.satts.stats()
        print(stats.sel(stats="median").values)
        assert np.array_equal(
            stats.sel(stats="median").values, np.array([[1.5, 5.5], [9.5, 13.5]])
        )

    def test_stats_returns_correct_mean(self, test_stack):
        stats = test_stack.satts.stats()
        assert np.array_equal(
            stats.sel(stats="mean").values, np.array([[1.5, 5.5], [9.5, 13.5]])
        )

    def test_stats_returns_correct_max(self, test_stack):
        stats = test_stack.satts.stats()
        assert np.array_equal(
            stats.sel(stats="max").values, np.array([[3, 7], [11, 15]])
        )

    def test_stats_returns_correct_min(self, test_stack):
        stats = test_stack.satts.stats()
        assert np.array_equal(
            stats.sel(stats="min").values, np.array([[0, 4], [8, 12]])
        )

    def test_stats_returns_correct_std(self, test_stack):
        stats = test_stack.satts.stats()

        assert (
            assert_almost_equal(
                stats.sel(stats="std").values,
                np.array([[1.11803399, 1.11803399], [1.11803399, 1.11803399]]),
            )
            is None
        )
