import pytest
import xarray as xr
import numpy as np
import rioxarray
import geopandas as gpd
import pandas as pd

from inspect import signature
from unittest.mock import patch, MagicMock, create_autospec
from numpy import testing as npt
from shapely.geometry import Polygon
from geopandas.testing import assert_geodataframe_equal

from tests.utils import SAME_XR

from spectral_recovery.restoration import (
    RestorationArea,
)
from spectral_recovery.enums import Metric
from spectral_recovery._config import DATETIME_FREQ

# TODO: move test data into their own folders, create temp dirs so individual tests
# don't conflict while reading the data
# https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data

TIMESERIES_LEN_17 = "src/tests/test_data/time17_xy2_epsg3005.tif"
TIMESERIES_LEN_3 = "src/tests/test_data/time3_xy2_epsg3005.tif"
POLYGON_INBOUND = "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
POLYGON_OUTBOUND = "src/tests/test_data/polygon_outbound_epsg3005.gpkg"
POLYGON_OVERLAP = "src/tests/test_data/polygon_overlap_epsg3005.gpkg"


def set_dates(
    gdf: gpd.GeoDataFrame,
    dist: int | str = None,
    rest: int | str = None,
    ref_start: int | str = None,
    ref_end: int | str = None,
):
    if dist:
        gdf["dist_year"] = dist
    if rest:
        gdf["rest_year"] = rest
    if ref_start:
        gdf["ref_start"] = ref_start
    if ref_end:
        gdf["ref_end"] = ref_end
    return gdf


class TestRestorationAreaInit:
    @pytest.mark.parametrize(
        (
            "resto_poly",
            "resto_start",
            "dist_start",
            "ref_years",
            "raster",
            "time_range",
        ),
        [
            (
                POLYGON_INBOUND,
                "2014",
                "2013",
                ["2010", "2011"],
                TIMESERIES_LEN_17,
                [str(x) for x in np.arange(2010, 2027)],
            ),
        ],
    )
    def test_good_init(
        self,
        resto_poly,
        resto_start,
        dist_start,
        ref_years,
        raster,
        time_range,
    ):
        with rioxarray.open_rasterio(raster, chunks="auto") as data:
            stack = data
            stack = stack.rename({"band": "time"})
            stack = stack.expand_dims(dim={"band": [0]})
            stack = stack.assign_coords(
                time=(pd.date_range(time_range[0], time_range[-1], freq=DATETIME_FREQ))
            )
            resto_poly = gpd.read_file(POLYGON_INBOUND)
            resto_poly = set_dates(
                resto_poly,
                dist=dist_start,
                rest=resto_start,
                ref_start=ref_years[0],
                ref_end=ref_years[1],
            )

            recovery_target = xr.DataArray(np.ones(stack.sizes["band"]), dims=["band"], coords={"band": [0]})

            resto_a = RestorationArea(
                restoration_site=resto_poly,
                composite_stack=stack,
                recovery_target=recovery_target
            )

            assert (
                resto_a.restoration_site.geometry.geom_equals(resto_poly.geometry)
            ).all()
            assert resto_a.disturbance_start == dist_start
            assert resto_a.restoration_start == resto_start
            assert resto_a.disturbance_start == dist_start
            assert resto_a.timeseries_start == time_range[0]
            assert resto_a.timeseries_end == time_range[-1]


class TestRestorationAreaComposite:
    def test_composite_stack_wrong_dims_throws_value_error(self):
        with rioxarray.open_rasterio(TIMESERIES_LEN_3) as data:
            # Misname band dimension to "bandz"
            bad_stack = data
            bad_stack = bad_stack.rename({"band": "time"})
            bad_stack = bad_stack.expand_dims(dim={"bandz": [0]})
            bad_stack = bad_stack.assign_coords(
                time=([
                    pd.to_datetime("2020"),
                    pd.to_datetime("2021"),
                    pd.to_datetime("2022"),
                ])
            )
            recovery_target = xr.DataArray(np.ones(bad_stack.sizes["bandz"]), dims=["bandz"], coords={"bandz": [0]})

            resto_poly = gpd.read_file(POLYGON_INBOUND)
            resto_poly = set_dates(
                resto_poly, dist="2020", rest="2021", ref_start="2019", ref_end="2019"
            )


            with pytest.raises(
                ValueError,
            ):
                resto_a = RestorationArea(
                    restoration_site=resto_poly,
                    composite_stack=bad_stack,
                    recovery_target=recovery_target,
                )

    def test_composite_stack_missing_dims_throws_value_error(self):
        with rioxarray.open_rasterio(TIMESERIES_LEN_3) as data:
            # Do not add a "band" dimension
            bad_stack = data
            bad_stack = bad_stack.rename({"band": "time"})
            bad_stack = bad_stack.assign_coords(
                time=([
                    pd.to_datetime("2020"),
                    pd.to_datetime("2021"),
                    pd.to_datetime("2023"),
                ])
            )
            recovery_target = xr.DataArray(np.ones(1), dims=["band"], coords={"band": [0]})

    
            resto_poly = gpd.read_file(POLYGON_INBOUND)
            resto_poly = set_dates(
                resto_poly, dist="2020", rest="2021", ref_start="2019", ref_end="2019"
            )

            with pytest.raises(
                ValueError,
            ):
                resto_a = RestorationArea(
                    restoration_site=resto_poly,
                    composite_stack=bad_stack,
                    recovery_target=recovery_target
                )

    def test_composite_stack_missing_years_throws_value_error(self):
        with rioxarray.open_rasterio(TIMESERIES_LEN_3) as data:
            # Set coordinates to 2020, 2021, and 2023, missing 2022
            bad_stack = data
            bad_stack = bad_stack.rename({"band": "time"})
            bad_stack = bad_stack.expand_dims(dim={"band": [0]})
            bad_stack = bad_stack.assign_coords(
                time=([
                    pd.to_datetime("2020"),
                    pd.to_datetime("2021"),
                    pd.to_datetime("2023"),
                ])
            )
            recovery_target = xr.DataArray(np.ones(bad_stack.sizes["band"]), dims=["band"], coords={"band": [0]})


            resto_poly = gpd.read_file(POLYGON_INBOUND)
            resto_poly = set_dates(
                resto_poly, dist="2020", rest="2021", ref_start="2019", ref_end="2019"
            )

            with pytest.raises(
                ValueError,
            ):
                resto_a = RestorationArea(
                    restoration_site=resto_poly,
                    composite_stack=bad_stack,
                    recovery_target=recovery_target
                )


class TestRestorationAreaPolygons:
    @pytest.fixture()
    def valid_timeseries(self):
        with rioxarray.open_rasterio(TIMESERIES_LEN_17, chunks="auto") as data:
            valid_timeseries = data.rename({"band": "time"}).expand_dims(
                dim={"band": [0]}
            )
            valid_timeseries = valid_timeseries.assign_coords(
                time=(pd.date_range("2010", "2026", freq=DATETIME_FREQ))
            )
        return valid_timeseries

    @pytest.mark.parametrize(
        ("polygon",),
        [
            (POLYGON_OUTBOUND,),  # bad spatial location (not contained at all)
            (POLYGON_OVERLAP,),  # bad spatial location (not fully contained)
        ],
    )
    def test_out_of_bounds_polygons_throw_value_err(self, polygon, valid_timeseries):
        resto_poly = gpd.read_file(polygon)
        resto_poly = set_dates(
            resto_poly, dist="2020", rest="2021", ref_start="2019", ref_end="2019"
        )
        recovery_target = xr.DataArray(np.ones(valid_timeseries.sizes["band"]), dims=["band"], coords={"band": 0})


        with pytest.raises(
            ValueError,
        ):
            RestorationArea(
                restoration_site=resto_poly,
                composite_stack=valid_timeseries,
                recovery_target=recovery_target,
            )

    def test_in_bounds_polygon_returns_same_polygon(self, valid_timeseries):
        resto_poly = gpd.read_file(POLYGON_INBOUND)
        resto_poly = set_dates(
            resto_poly, dist="2020", rest="2021", ref_start="2019", ref_end="2019"
        )
        recovery_target = xr.DataArray(np.ones(valid_timeseries.sizes["band"]), dims=["band"], coords={"band": [0]})


        ra = RestorationArea(
            restoration_site=resto_poly,
            composite_stack=valid_timeseries,
            recovery_target=recovery_target
        )

        assert_geodataframe_equal(ra.restoration_site, resto_poly)

    def test_gdf_with_nonzero_row_index_does_not_fail(self, valid_timeseries):
        resto_poly = gpd.read_file(POLYGON_INBOUND)
        non_zero_row = resto_poly.rename(index={0: 20})
        non_zero_row = set_dates(
            non_zero_row, dist="2020", rest="2021", ref_start="2019", ref_end="2019"
        )
        recovery_target = xr.DataArray(np.ones(valid_timeseries.sizes["band"]), dims=["band"], coords={"band": [0]})


        RestorationArea(
            restoration_site=non_zero_row,
            composite_stack=valid_timeseries,
            recovery_target=recovery_target,
        )


class TestRestorationAreaDates:
    @pytest.fixture()
    def valid_array(self):
        data = np.ones((1, 5, 2, 2))
        bands = [0]
        latitudes = [0, 1]
        longitudes = [0, 1]
        time = pd.date_range("2010", "2014", freq="YS")
        xarr = xr.DataArray(
            data,
            dims=["band", "time", "y", "x"],
            coords={"band": bands, "time": time, "y": latitudes, "x": longitudes},
        )
        xarr.rio.write_crs("EPSG:4326", inplace=True)
        return xarr

    @pytest.fixture()
    def valid_poly(self):
        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        return polygon
    
    @pytest.fixture()
    def valid_target(self, valid_array):
        valid_target = xr.DataArray(np.ones(valid_array.sizes["band"]), dims=["band"], coords={"band": [0]})
        return valid_target

    def test_str_dates_from_int(self, valid_array, valid_poly, valid_target):
        poly_frame_int_dates = gpd.GeoDataFrame(
            {
                "dist_start": [2012],
                "rest_start": [2013],
                "geometry": [valid_poly],
            },
            crs="EPSG:4326",
        )

        ra = RestorationArea(
            composite_stack=valid_array,
            restoration_site=poly_frame_int_dates,
            recovery_target=valid_target,
        )

        assert isinstance(ra.disturbance_start, str)
        assert isinstance(ra.restoration_start, str)

    def test_returns_str_dates_from_str(self, valid_array, valid_poly, valid_target):
        poly_frame_str_dates = gpd.GeoDataFrame(
            {
                "dist_start": ["2012"],
                "rest_start": ["2013"],
                "geometry": [valid_poly],
            },
            crs="EPSG:4326",
        )
        ra = RestorationArea(
            composite_stack=valid_array,
            restoration_site=poly_frame_str_dates,
            recovery_target=valid_target,
        )

        assert isinstance(ra.disturbance_start, str)
        assert isinstance(ra.restoration_start, str)

    def test_missing_dates_throws_value_err(self, valid_array, valid_poly, valid_target):
        frame_missing_dates = gpd.GeoDataFrame(
            {
                "rest_start": [2013],
                "geometry": [valid_poly],
            },
            crs="EPSG:4326",
        )

        with pytest.raises(ValueError):
            RestorationArea(
                restoration_site=frame_missing_dates,
                composite_stack=valid_array,
                recovery_target=valid_target,
            )

    def test_dist_year_greater_than_rest_year_throws_value_error(
        self, valid_array, valid_poly, valid_target
    ):
        dist_greater_rest = gpd.GeoDataFrame(
            {
                "dist_start": [2012],
                "rest_start": [2011],
                "geometry": [valid_poly],
            },
            crs="EPSG:4326",
        )

        with pytest.raises(ValueError):
            RestorationArea(
                restoration_site=dist_greater_rest,
                composite_stack=valid_array,
                recovery_target=valid_target,
            )

    def test_out_of_bounds_restoration_start_throws_value_error(
        self, valid_array, valid_poly, valid_target
    ):
        oob_rest = gpd.GeoDataFrame(
            {
                "dist_start": [2011],
                "rest_start": [2015],
                "geometry": [valid_poly],
            },
            crs="EPSG:4326",
        )

        with pytest.raises(ValueError):
            RestorationArea(
                restoration_site=oob_rest,
                composite_stack=valid_array,
                recovery_target=valid_target,
            )

    def test_out_of_bounds_disturbance_start_year_throws_value_error(
        self, valid_array, valid_poly, valid_target
    ):
        oob_dist = gpd.GeoDataFrame(
            {
                "dist_start": [2008],
                "rest_start": [2012],
                "geometry": [valid_poly],
            },
            crs="EPSG:4326",
        )

        with pytest.raises(ValueError):
            RestorationArea(
                restoration_site=oob_dist,
                composite_stack=valid_array,
                recovery_target=valid_target,
            )