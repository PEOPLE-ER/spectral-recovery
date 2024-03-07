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

from spectral_recovery.targets import MedianTarget
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

            resto_a = RestorationArea(
                restoration_polygon=resto_poly,
                composite_stack=stack,
            )

            assert (
                resto_a.restoration_polygon.geometry.geom_equals(resto_poly.geometry)
            ).all()
            assert resto_a.reference_polygons == None
            assert resto_a.disturbance_start == dist_start
            assert resto_a.restoration_start == resto_start
            assert resto_a.disturbance_start == dist_start
            assert resto_a.reference_years == ref_years
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

            resto_poly = gpd.read_file(POLYGON_INBOUND)
            resto_poly = set_dates(
                resto_poly, dist="2020", rest="2021", ref_start="2019", ref_end="2019"
            )

            with pytest.raises(
                ValueError,
            ):
                resto_a = RestorationArea(
                    restoration_polygon=resto_poly,
                    reference_polygons=resto_poly,
                    composite_stack=bad_stack,
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

            resto_poly = gpd.read_file(POLYGON_INBOUND)
            resto_poly = set_dates(
                resto_poly, dist="2020", rest="2021", ref_start="2019", ref_end="2019"
            )

            with pytest.raises(
                ValueError,
            ):
                resto_a = RestorationArea(
                    restoration_polygon=resto_poly,
                    reference_polygons=resto_poly,
                    composite_stack=bad_stack,
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

            resto_poly = gpd.read_file(POLYGON_INBOUND)
            resto_poly = set_dates(
                resto_poly, dist="2020", rest="2021", ref_start="2019", ref_end="2019"
            )

            with pytest.raises(
                ValueError,
            ):
                resto_a = RestorationArea(
                    restoration_polygon=resto_poly,
                    composite_stack=bad_stack,
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

        with pytest.raises(
            ValueError,
        ):
            RestorationArea(
                restoration_polygon=resto_poly,
                composite_stack=valid_timeseries,
            )

    def test_in_bounds_polygon_returns_same_polygon(self, valid_timeseries):
        resto_poly = gpd.read_file(POLYGON_INBOUND)
        resto_poly = set_dates(
            resto_poly, dist="2020", rest="2021", ref_start="2019", ref_end="2019"
        )

        ra = RestorationArea(
            restoration_polygon=resto_poly,
            composite_stack=valid_timeseries,
        )

        assert_geodataframe_equal(ra.restoration_polygon, resto_poly)


class TestValidateReferencePolygons:
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
        rest_poly = gpd.read_file(POLYGON_INBOUND)
        rest_poly = set_dates(rest_poly, dist="2020", rest="2021")

        ref_poly = gpd.read_file(polygon)
        ref_poly = set_dates(ref_poly, ref_start="2019", ref_end="2019")

        with pytest.raises(
            ValueError,
        ):
            RestorationArea(
                restoration_polygon=rest_poly,
                reference_polygons=ref_poly,
                composite_stack=valid_timeseries,
            )

    def test_in_bounds_polygon_returns_same_polygon(self, valid_timeseries):
        rest_poly = gpd.read_file(POLYGON_INBOUND)
        rest_poly = set_dates(rest_poly, dist="2020", rest="2021")

        ref_poly = gpd.read_file(POLYGON_INBOUND)
        ref_poly = set_dates(ref_poly, ref_start="2019", ref_end="2019")

        ra = RestorationArea(
            restoration_polygon=rest_poly,
            reference_polygons=rest_poly,
            composite_stack=valid_timeseries,
        )
        assert_geodataframe_equal(ra.reference_polygons, rest_poly)

    def test_ref_not_geodataframe_polygon_throws_value_err(self, valid_timeseries):
        rest_poly = gpd.read_file(POLYGON_INBOUND)
        rest_poly = set_dates(rest_poly, dist="2020", rest="2021")

        ref_poly = None

        with pytest.raises(ValueError):
            RestorationArea(
                restoration_polygon=rest_poly,
                reference_polygons=ref_poly,
                composite_stack=valid_timeseries,
            )


class TestRestorationAreaDates:
    @pytest.fixture()
    def valid_array(self):
        data = np.ones((1, 5, 2, 2))
        latitudes = [0, 1]
        longitudes = [0, 1]
        time = pd.date_range("2010", "2014", freq="YS")
        xarr = xr.DataArray(
            data,
            dims=["band", "time", "y", "x"],
            coords={"time": time, "y": latitudes, "x": longitudes},
        )
        xarr.rio.write_crs("EPSG:4326", inplace=True)
        return xarr

    @pytest.fixture()
    def valid_poly(self):
        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        return polygon

    def test_str_dates_from_int(self, valid_array, valid_poly):
        poly_frame_int_dates = gpd.GeoDataFrame(
            {
                "dist_start": [2012],
                "rest_start": [2013],
                "reference_start": [2010],
                "reference_end": [2010],
                "geometry": [valid_poly],
            },
            crs="EPSG:4326",
        )

        ra = RestorationArea(
            composite_stack=valid_array,
            restoration_polygon=poly_frame_int_dates,
        )

        assert isinstance(ra.disturbance_start, str)
        assert isinstance(ra.restoration_start, str)
        assert isinstance(ra.reference_years[0], str)
        assert isinstance(ra.reference_years[1], str)

    def test_returns_str_dates_from_str(self, valid_array, valid_poly):
        poly_frame_str_dates = gpd.GeoDataFrame(
            {
                "dist_start": ["2012"],
                "rest_start": ["2013"],
                "reference_start": ["2010"],
                "reference_end": ["2010"],
                "geometry": [valid_poly],
            },
            crs="EPSG:4326",
        )
        ra = RestorationArea(
            composite_stack=valid_array,
            restoration_polygon=poly_frame_str_dates,
        )

        assert isinstance(ra.disturbance_start, str)
        assert isinstance(ra.restoration_start, str)
        assert isinstance(ra.reference_years[0], str)
        assert isinstance(ra.reference_years[1], str)

    def test_ref_none_takes_all_dates_from_rest(self, valid_array, valid_poly):
        full_rest_frame = gpd.GeoDataFrame(
            {
                "dist_start": [2012],
                "rest_start": [2013],
                "reference_start": [2010],
                "reference_end": [2010],
                "geometry": [valid_poly],
            },
            crs="EPSG:4326",
        )

        ra = RestorationArea(
            composite_stack=valid_array,
            restoration_polygon=full_rest_frame,
        )

        assert ra.disturbance_start == "2012"
        assert ra.restoration_start == "2013"
        assert ra.reference_years[0] == "2010"
        assert ra.reference_years[1] == "2010"

    def test_ref_dates_take_from_ref(self, valid_array, valid_poly):
        rest_frame_no_ref = gpd.GeoDataFrame(
            {"dist_start": [2012], "rest_start": [2013], "geometry": [valid_poly]},
            crs="EPSG:4326",
        )
        full_ref_frame = gpd.GeoDataFrame(
            {"ref_start": [2010], "ref_end": [2010], "geometry": [valid_poly]},
            crs="EPSG:4326",
        )
        ra = RestorationArea(
            composite_stack=valid_array,
            restoration_polygon=rest_frame_no_ref,
            reference_polygons=full_ref_frame,
        )

        assert ra.disturbance_start == "2012"
        assert ra.restoration_start == "2013"
        assert ra.reference_years[0] == "2010"
        assert ra.reference_years[1] == "2010"

    def test_ref_dates_override_rest_dates(self, valid_array, valid_poly):
        full_rest_frame = gpd.GeoDataFrame(
            {
                "dist_start": [2012],
                "rest_start": [2013],
                "ref_start": [2011],
                "ref_end": [2011],
                "geometry": [valid_poly],
            },
            crs="EPSG:4326",
        )
        full_ref_frame = gpd.GeoDataFrame(
            {"ref_start": [2010], "ref_end": [2010], "geometry": [valid_poly]},
            crs="EPSG:4326",
        )

        ra = RestorationArea(
            composite_stack=valid_array,
            restoration_polygon=full_rest_frame,
            reference_polygons=full_ref_frame,
        )

        assert ra.disturbance_start == "2012"
        assert ra.restoration_start == "2013"
        assert ra.reference_years[0] == "2010"
        assert ra.reference_years[1] == "2010"

    def test_missing_dates_throws_value_err(self, valid_array, valid_poly):
        frame_missing_dates = gpd.GeoDataFrame(
            {
                "rest_start": [2013],
                "reference_start": [2010],
                "reference_end": [2010],
                "geometry": [valid_poly],
            },
            crs="EPSG:4326",
        )

        with pytest.raises(ValueError):
            RestorationArea(
                restoration_polygon=frame_missing_dates,
                reference_polygons=None,
                composite_stack=valid_array,
            )

    def test_dist_year_greater_than_rest_year_throws_value_error(
        self, valid_array, valid_poly
    ):
        dist_greater_rest = gpd.GeoDataFrame(
            {
                "dist_start": [2012],
                "rest_start": [2011],
                "reference_start": [2010],
                "reference_end": [2010],
                "geometry": [valid_poly],
            },
            crs="EPSG:4326",
        )

        with pytest.raises(ValueError):
            RestorationArea(
                restoration_polygon=dist_greater_rest,
                reference_polygons=None,
                composite_stack=valid_array,
            )

    def test_out_of_bounds_restoration_start_throws_value_error(
        self, valid_array, valid_poly
    ):
        oob_rest = gpd.GeoDataFrame(
            {
                "dist_start": [2011],
                "rest_start": [2015],
                "reference_start": [2010],
                "reference_end": [2010],
                "geometry": [valid_poly],
            },
            crs="EPSG:4326",
        )

        with pytest.raises(ValueError):
            RestorationArea(
                restoration_polygon=oob_rest,
                reference_polygons=None,
                composite_stack=valid_array,
            )

    def test_out_of_bounds_disturbance_start_year_throws_value_error(
        self, valid_array, valid_poly
    ):
        oob_dist = gpd.GeoDataFrame(
            {
                "dist_start": [2008],
                "rest_start": [2012],
                "reference_start": [2010],
                "reference_end": [2010],
                "geometry": [valid_poly],
            },
            crs="EPSG:4326",
        )

        with pytest.raises(ValueError):
            RestorationArea(
                restoration_polygon=oob_dist,
                reference_polygons=None,
                composite_stack=valid_array,
            )

    @pytest.mark.parametrize(
        ("ref_year_start", "ref_year_end"),
        [
            (2002, 2002),
            (2025, 2028),
            (2008, 2011),
        ],
    )
    def test_out_of_bounds_reference_years_throw_value_error(
        self, valid_array, valid_poly, ref_year_start, ref_year_end
    ):
        oob_ref = gpd.GeoDataFrame(
            {
                "dist_start": [2008],
                "rest_start": [2015],
                "reference_start": [ref_year_start],
                "reference_end": [ref_year_end],
                "geometry": [valid_poly],
            },
            crs="EPSG:4326",
        )

        with pytest.raises(ValueError):
            RestorationArea(
                restoration_polygon=oob_ref,
                reference_polygons=None,
                composite_stack=valid_array,
            )


class TestRestorationAreaRecoveryTarget:
    @pytest.fixture()
    def valid_ra_build(self):
        # TODO: Simplify this to just use int coords and polygons that intersect. Shouldn't need to read the files.
        resto_poly = gpd.read_file(POLYGON_INBOUND)
        resto_poly["dist_start"] = "2015"
        resto_poly["rest_start"] = "2016"
        resto_poly["ref_start"] = "2010"
        resto_poly["ref_end"] = "2010"

        raster = TIMESERIES_LEN_17
        time_range = [str(x) for x in np.arange(2010, 2027)]

        with rioxarray.open_rasterio(raster, chunks="auto") as data:
            stack = data
            stack = stack.rename({"band": "time"})
            stack = stack.expand_dims(dim={"band": [0]})
            stack = stack.assign_coords(
                time=(pd.date_range(time_range[0], time_range[-1], freq=DATETIME_FREQ))
            )

        return {
            "restoration_polygon": resto_poly,
            "composite_stack": stack,
        }

    def test_default_is_median_target_instance_w_scale_polygon(
        self,
        valid_ra_build,
    ):
        resto_a = RestorationArea(**valid_ra_build)

        assert resto_a.recovery_target_method.scale == "polygon"

    def test_recovery_target_method_with_valid_sig_calls_once(
        self,
        valid_ra_build,
    ):
        valid_method = create_autospec(MedianTarget(scale="polygon"))

        resto_a = RestorationArea(**valid_ra_build, recovery_target_method=valid_method)
        resto_a.recovery_target  # Trigger computation

        valid_method.assert_called_once()

    def test_recovery_target_method_with_invalid_sig_throws_value_error(
        self, valid_ra_build
    ):
        def invalid_method(a: int, b: str):
            """Method with incorrect signature."""
            return 0

        with pytest.raises(ValueError):
            resto_a = RestorationArea(
                **valid_ra_build, recovery_target_method=invalid_method
            )

    def test_pixel_recovery_target_with_reference_polygons_throws_type_error(
        self, valid_ra_build
    ):
        reference_frame = gpd.GeoDataFrame({
            "ref_start": 2010,
            "ref_end": 2011,
            "geometry": valid_ra_build["restoration_polygon"].geometry.values,
        })
        valid_ra_build["reference_polygons"] = reference_frame
        median_pixel = MedianTarget(scale="pixel")

        with pytest.raises(TypeError):
            ra = RestorationArea(**valid_ra_build, recovery_target_method=median_pixel)
