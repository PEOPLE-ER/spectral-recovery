import pytest
import xarray as xr
import numpy as np
import rioxarray
import geopandas as gpd
import pandas as pd

from inspect import signature
from unittest.mock import patch, MagicMock, create_autospec
from numpy import testing as npt
from geopandas.testing import assert_geodataframe_equal
from tests.utils import SAME_XR

from spectral_recovery.targets import MedianTarget
from spectral_recovery.restoration import (

    _validate_dates,
    _validate_restoration_polygons,
    _validate_reference_polygons,
    _get_dates_from_frame,
    RestorationArea,
)
from spectral_recovery.enums import Metric
from spectral_recovery._config import DATETIME_FREQ

# TODO: move test data into their own folders, create temp dirs so individual tests
# don't conflict while reading the data
# https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data


class TestRestorationAreaInit:
    @pytest.mark.parametrize(
        ("resto_poly", "resto_start", "dist_start", "ref_years", "raster", "time_range"),
        [
            (
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg",
                "2014",
                "2013",
                ["2010", "2010"],
                "src/tests/test_data/time17_xy2_epsg3005.tif",
                [str(x) for x in np.arange(2010, 2027)],
            ),
            (
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg",
                "2014",
                "2013",
                ["2010", "2011"],
                "src/tests/test_data/time17_xy2_epsg3005.tif",
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
            resto_poly = gpd.read_file(
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
            )
            resto_poly["dist_year"] = dist_start
            resto_poly["rest_year"] = resto_start
            resto_poly["ref_start"] = ref_years[0]
            resto_poly["ref_end"] = ref_years[1]

            print(resto_poly)


            resto_a = RestorationArea(
                restoration_polygon=resto_poly,
                composite_stack=stack,
            )

            assert (
                resto_a.restoration_polygon.geometry.geom_equals(resto_poly.geometry)
            ).all()
            assert resto_a.restoration_start == resto_start
            assert resto_a.reference_years == ref_years


    def test_composite_stack_wrong_dims_throws_value_error(self):
        with rioxarray.open_rasterio(
            "src/tests/test_data/time3_xy2_epsg3005.tif"
        ) as data:
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
            dist_start = "2020"
            resto_start = "2021"
            ref_years = "2019"
            resto_poly = gpd.read_file(
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
            )
            resto_poly["dist_year"] = dist_start
            resto_poly["rest_year"] = resto_start
            resto_poly["ref_start"] = ref_years
            resto_poly["ref_end"] = ref_years

            with pytest.raises(
                ValueError,
            ):
                resto_a = RestorationArea(
                    restoration_polygon=resto_poly,
                    reference_polygons=resto_poly,
                    composite_stack=bad_stack,
                )

    def test_composite_stack_missing_dims_throws_value_error(self):
        with rioxarray.open_rasterio(
            "src/tests/test_data/time3_xy2_epsg3005.tif"
        ) as data:
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
            dist_start = "2020"
            resto_start = "2021"
            ref_years = "2019"
            resto_poly = gpd.read_file(
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
            )
            resto_poly["dist_year"] = dist_start
            resto_poly["rest_year"] = resto_start
            resto_poly["ref_start"] = ref_years
            resto_poly["ref_end"] = ref_years

            with pytest.raises(
                ValueError,
            ):
                resto_a = RestorationArea(
                    restoration_polygon=resto_poly,
                    reference_polygons=resto_poly,
                    composite_stack=bad_stack,
                )

    def test_composite_stack_missing_years_throws_value_error(self):
        with rioxarray.open_rasterio(
            "src/tests/test_data/time3_xy2_epsg3005.tif"
        ) as data:
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
            dist_start = "2020"
            resto_start = "2021"
            ref_years = "2019"
            resto_poly = gpd.read_file(
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
            )
            resto_poly["dist_year"] = dist_start
            resto_poly["rest_year"] = resto_start
            resto_poly["ref_start"] = ref_years
            resto_poly["ref_end"] = ref_years

            with pytest.raises(
                ValueError,
            ):
                resto_a = RestorationArea(
                    restoration_polygon=resto_poly,
                    composite_stack=bad_stack,
                )


class TestValidateRestorationPolygons:
    @pytest.mark.parametrize(
        (
            "polygon",
            "raster",
        ),
        [
            (  # bad spatial location (not contained at all)
                "src/tests/test_data/polygon_outbound_epsg3005.gpkg",
                "src/tests/test_data/time17_xy2_epsg3005.tif",
            ),
            (  # bad spatial location (not fully contained)
                "src/tests/test_data/polygon_overlap_epsg3005.gpkg",
                "src/tests/test_data/time17_xy2_epsg3005.tif",
            ),
        ],
    )
    def test_out_of_bounds_polygons_throw_value_err(
        self, polygon, raster
    ):
        with rioxarray.open_rasterio(raster, chunks="auto") as data:
            stack = data.rename({"band": "time"}).expand_dims(dim={"band": [0]})
            stack = stack.assign_coords(
                time=(pd.date_range("2010", "2026", freq=DATETIME_FREQ))
            )
            resto_poly = gpd.read_file(polygon)

            with pytest.raises(
                ValueError,
            ):
                rest = _validate_restoration_polygons(
                    restoration_polygon=resto_poly,
                    image_stack=stack,
                )
    
    def test_in_bounds_polygon_returns_same_polygon(self):
        with rioxarray.open_rasterio("src/tests/test_data/time17_xy2_epsg3005.tif", chunks="auto") as data:
            stack = data.rename({"band": "time"}).expand_dims(dim={"band": [0]})
            stack = stack.assign_coords(
                time=(pd.date_range("2010", "2026", freq=DATETIME_FREQ))
            )
            resto_poly = gpd.read_file("src/tests/test_data/polygon_inbound_epsg3005.gpkg")

            rp = _validate_restoration_polygons(
                    restoration_polygon=resto_poly,
                    image_stack=stack,
                )
        
        assert_geodataframe_equal(rp, resto_poly)


class TestValidateReferencePolygons: 
    @pytest.mark.parametrize(
        (
            "polygon",
            "raster",
        ),
        [
            (  # bad spatial location (not contained at all)
                "src/tests/test_data/polygon_outbound_epsg3005.gpkg",
                "src/tests/test_data/time17_xy2_epsg3005.tif",
            ),
            (  # bad spatial location (not fully contained)
                "src/tests/test_data/polygon_overlap_epsg3005.gpkg",
                "src/tests/test_data/time17_xy2_epsg3005.tif",
            ),
        ],
    )
    def test_out_of_bounds_polygons_throw_value_err(
        self, polygon, raster
    ):
        with rioxarray.open_rasterio(raster, chunks="auto") as data:
            stack = data.rename({"band": "time"}).expand_dims(dim={"band": [0]})
            stack = stack.assign_coords(
                time=(pd.date_range("2010", "2026", freq=DATETIME_FREQ))
            )
            ref_poly = gpd.read_file(polygon)

            with pytest.raises(
                ValueError,
            ):
                rest = _validate_reference_polygons(
                    reference_polygons=ref_poly,
                    image_stack=stack,
                )
    
    def test_in_bounds_polygon_returns_same_polygon(self):
        with rioxarray.open_rasterio("src/tests/test_data/time17_xy2_epsg3005.tif", chunks="auto") as data:
            stack = data.rename({"band": "time"}).expand_dims(dim={"band": [0]})
            stack = stack.assign_coords(
                time=(pd.date_range("2010", "2026", freq=DATETIME_FREQ))
            )
            ref_poly = gpd.read_file("src/tests/test_data/polygon_inbound_epsg3005.gpkg")

            rp = _validate_reference_polygons(
                    reference_polygons=ref_poly,
                    image_stack=stack,
                )
        
        assert_geodataframe_equal(rp, ref_poly)
    
    def test_none_polygon_returns_none(self):
        with rioxarray.open_rasterio("src/tests/test_data/time17_xy2_epsg3005.tif", chunks="auto") as data:
            stack = data.rename({"band": "time"}).expand_dims(dim={"band": [0]})
            stack = stack.assign_coords(
                time=(pd.date_range("2010", "2026", freq=DATETIME_FREQ))
            )
            ref_poly = None

            rp = _validate_reference_polygons(
                    reference_polygons=ref_poly,
                    image_stack=stack,
                )
        
        assert rp == None

class TestDatesFromFrames:

    def test_returns_str_dates_from_int(self):
        full_rest =  gpd.GeoDataFrame({'dist_start': [2012], 'rest_start': [2013], "reference_start": [2010], "reference_end": [2010], 'geometry': ['POINT (1 2)']})
        dist, rest, ref = _get_dates_from_frame(rest_frame=full_rest, ref_frame=None)

        assert isinstance(dist, str)
        assert isinstance(rest, str)
        assert isinstance(ref[0], str)
        assert isinstance(ref[1], str)
    
    def test_returns_str_dates_from_str(self):
        full_rest =  gpd.GeoDataFrame({'dist_start': ["2012"], 'rest_start': ["2013"], "reference_start": ["2010"], "reference_end": ["2010"], 'geometry': ['POINT (1 2)']})
        dist, rest, ref = _get_dates_from_frame(rest_frame=full_rest, ref_frame=None)

        assert isinstance(dist, str)
        assert isinstance(rest, str)
        assert isinstance(ref[0], str)
        assert isinstance(ref[1], str)

    def test_ref_none_takes_all_dates_from_rest(self):
        full_rest =  gpd.GeoDataFrame({'dist_start': [2012], 'rest_start': [2013], "reference_start": [2010], "reference_end": [2010], 'geometry': ['POINT (1 2)']})
        dist, rest, ref = _get_dates_from_frame(rest_frame=full_rest, ref_frame=None)

        assert dist == "2012"
        assert rest == "2013"
        assert ref[0] == "2010"
        assert ref[1] == "2010"
    
    def test_ref_from_ref(self):
        full_rest =  gpd.GeoDataFrame({'dist_start': [2012], 'rest_start': [2013], 'geometry': ['POINT (1 2)']})
        full_ref = gpd.GeoDataFrame({'ref_start': [2010], 'ref_end': [2010], 'geometry': ['POINT (1 2)']})
        dist, rest, ref = _get_dates_from_frame(rest_frame=full_rest, ref_frame=full_ref)

        assert dist == "2012"
        assert rest == "2013"
        assert ref[0] == "2010"
        assert ref[1] == "2010"

class TestValidateDates:

    def test_missing_dates_throws_value_err(self):

        test_stack = xr.DataArray(
            np.ones((1, 3, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"time": pd.date_range("2009", "2011", freq=DATETIME_FREQ)},
        )

        frame_missing_dates =  gpd.GeoDataFrame({'rest_start': [2013], "reference_start": [2010], "reference_end": [2010], 'geometry': ['POINT (1 2)']})
        with pytest.raises(ValueError):

            ref, dist, rest = _validate_dates(
                rest_frame=frame_missing_dates,
                ref_frame=None,
                image_stack=test_stack,
            )

    def test_dist_year_greater_than_rest_year_throws_value_error(self):
        test_stack = xr.DataArray(
            np.ones((1, 3, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"time": pd.date_range("2009", "2011", freq=DATETIME_FREQ)},
        )

        dist_greater_rest =  gpd.GeoDataFrame({'dist_start': [2011], 'rest_start': [2010], "reference_start": [2009], "reference_end": [2009], 'geometry': ['POINT (1 2)']})

        with pytest.raises(ValueError):
            ref, dist, rest = _validate_dates(
                rest_frame=dist_greater_rest,
                ref_frame=None,
                image_stack=test_stack,
            )

    def test_out_of_bounds_restoration_start_year_throws_value_error(self):
        test_stack = xr.DataArray(
            np.ones((1, 3, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"time": pd.date_range("2009", "2011", freq=DATETIME_FREQ)},
        )
        oob_rest =  gpd.GeoDataFrame({'dist_start': [2011], 'rest_start': [2015], "reference_start": [2009], "reference_end": [2009], 'geometry': ['POINT (1 2)']})

        with pytest.raises(ValueError):
            ref, dist, rest = _validate_dates(
                rest_frame=oob_rest,
                ref_frame=None,
                image_stack=test_stack,
            )

    def test_out_of_bounds_disturbance_start_year_throws_value_error(self):
        test_stack = xr.DataArray(
            np.ones((1, 3, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"time": pd.date_range("2009", "2011", freq=DATETIME_FREQ)},
        )
        oob_dist =  gpd.GeoDataFrame({'dist_start': [2008], 'rest_start': [2015], "reference_start": [2009], "reference_end": [2009], 'geometry': ['POINT (1 2)']})

        with pytest.raises(ValueError):
            ref, dist, rest = _validate_dates(
                rest_frame=oob_dist,
                ref_frame=None,
                image_stack=test_stack,
            )

    @pytest.mark.parametrize(
        ("ref_year_start", "ref_year_end"),
        [
            (2002, 2002),
            (2025, 2028),
            (2008, 2011),
        ],
    )
    def test_out_of_bounds_reference_years_throw_value_error(self, ref_year_start, ref_year_end):
        test_stack = xr.DataArray(
            np.ones((1, 3, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"time": pd.date_range("2009", "2011", freq=DATETIME_FREQ)},
        )
        oob_ref =  gpd.GeoDataFrame({'dist_start': [2008], 'rest_start': [2015], "reference_start": [ref_year_start], "reference_end": [ref_year_end], 'geometry': ['POINT (1 2)']})

        with pytest.raises(ValueError):
            ref, dist, rest = _validate_dates(
                rest_frame=oob_ref,
                ref_frame=None,
                image_stack=test_stack,
            )


class TestRestorationAreaRecoveryTarget:
    @pytest.fixture()
    def valid_ra_build(self):
        # TODO: Simplify this to just use int coords and polygons that intersect. Shouldn't need to read the files.
        resto_poly = gpd.read_file("src/tests/test_data/polygon_inbound_epsg3005.gpkg")
        resto_poly["dist_start"] = "2015"
        resto_poly["rest_start"] = "2016"
        resto_poly["ref_start"] = "2010"
        resto_poly["ref_end"] = "2010"

        raster = "src/tests/test_data/time17_xy2_epsg3005.tif"
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
        reference_frame = gpd.GeoDataFrame({"ref_start": 2010, "ref_end": 2011, 'geometry': valid_ra_build["restoration_polygon"].geometry.values})
        valid_ra_build["reference_polygons"] = reference_frame
        median_pixel = MedianTarget(scale="pixel")

        with pytest.raises(TypeError):
            ra = RestorationArea(**valid_ra_build, recovery_target_method=median_pixel)


class TestRestorationAreaMetrics:

    time_range = [str(x) for x in np.arange(2010, 2027)]
    baseline_array = xr.DataArray([[[1.0]], [[2.0]]])

    @pytest.fixture()
    def valid_resto_area(self):
        polygon = "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
        raster = "src/tests/test_data/time17_xy2_epsg3005.tif"

        with rioxarray.open_rasterio(raster, chunks="auto") as data:
            resto_poly = gpd.read_file(polygon)
            resto_poly["dist_start"] = "2014"
            resto_poly["rest_start"] = "2015"
            resto_poly["ref_start"] = "2011"
            resto_poly["ref_end"] = "2011"

            stack = data
            stack = stack.rename({"band": "time"})
            stack = stack.expand_dims(dim={"band": [0]})
            stack = stack.assign_coords(
                time=(
                    pd.date_range(
                        self.time_range[0], self.time_range[-1], freq=DATETIME_FREQ
                    )
                )
            )
            stack = xr.concat([stack, stack], dim=pd.Index([0, 1], name="band"))
            resto_area = RestorationArea(
                restoration_polygon=resto_poly,
                composite_stack=stack,
            )

            mock_target_return = self.baseline_array
            resto_area.recovery_target = self.baseline_array

        return resto_area

    @patch(
        "spectral_recovery.metrics.y2r",
    )
    def test_Y2R_call_default(self, method_mock, valid_resto_area):
        mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
        method_mock.return_value = mocked_return

        result = valid_resto_area.y2r()
        expected_result = mocked_return.expand_dims(dim={"metric": [str(Metric.Y2R)]})

        assert result.equals(expected_result)

        post_restoration = valid_resto_area.stack.sel(
            time=slice(valid_resto_area.restoration_start, None)
        )
        rest_start = valid_resto_area.restoration_start
        rest_end = str(valid_resto_area.end_year.year)
        default_percent = 80

        method_mock.assert_called_with(
            image_stack=SAME_XR(post_restoration),
            recovery_target=SAME_XR(self.baseline_array),
            rest_start=rest_start,
            percent=default_percent,
        )

    @patch(
        "spectral_recovery.metrics.yryr",
    )
    def test_YrYr_call_default(self, method_mock, valid_resto_area):
        mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
        method_mock.return_value = mocked_return

        result = valid_resto_area.yryr()
        expected_result = mocked_return.expand_dims(dim={"metric": [str(Metric.YRYR)]})

        assert result.equals(expected_result)

        timestep_default = 5

        method_mock.assert_called_with(
            image_stack=SAME_XR(valid_resto_area.stack),
            rest_start=valid_resto_area.restoration_start,
            timestep=timestep_default,
        )

    @patch(
        "spectral_recovery.metrics.dnbr",
    )
    def test_dNBR_call_default(self, method_mock, valid_resto_area):
        mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
        method_mock.return_value = mocked_return

        result = valid_resto_area.dnbr()
        expected_result = mocked_return.expand_dims(dim={"metric": [str(Metric.DNBR)]})

        assert result.equals(expected_result)

        timestep_default = 5

        method_mock.assert_called_with(
            image_stack=SAME_XR(valid_resto_area.stack),
            rest_start=valid_resto_area.restoration_start,
            timestep=timestep_default,
        )

    @patch(
        "spectral_recovery.metrics.rri",
    )
    def test_RRI_call_default(self, method_mock, valid_resto_area):
        mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
        method_mock.return_value = mocked_return

        result = valid_resto_area._rri()
        expected_result = mocked_return.expand_dims(dim={"metric": [str(Metric.RRI)]})

        assert result.equals(expected_result)

        timestep_default = 5

        method_mock.assert_called_with(
            image_stack=SAME_XR(valid_resto_area.stack),
            rest_start=valid_resto_area.restoration_start,
            dist_start=valid_resto_area.disturbance_start,
            timestep=timestep_default,
        )

    @patch(
        "spectral_recovery.metrics.r80p",
    )
    def test_R80P_call_default(self, method_mock, valid_resto_area):
        mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
        method_mock.return_value = mocked_return

        result = valid_resto_area.r80p()
        expected_result = mocked_return.expand_dims(dim={"metric": [str(Metric.R80P)]})

        assert result.equals(expected_result)

        percent_default = 80
        timestep_default = 5

        method_mock.assert_called_with(
            image_stack=SAME_XR(valid_resto_area.stack),
            rest_start=valid_resto_area.restoration_start,
            recovery_target=SAME_XR(self.baseline_array),
            timestep=timestep_default,
            percent=percent_default,
        )