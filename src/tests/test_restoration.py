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
    _get_reference_image_stack,
    _validate_dates,
    _validate_restoration_polygons,
    _validate_reference_polygons,
    RestorationArea,
)
from spectral_recovery.enums import Metric
from spectral_recovery._config import DATETIME_FREQ

# TODO: move test data into their own folders, create temp dirs so individual tests
# don't conflict while reading the data
# https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data


class TestRestorationAreaInit:
    @pytest.mark.parametrize(
        ("resto_poly", "resto_start", "ref_years", "raster", "time_range"),
        [
            (
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg",
                "2014",
                "2010",
                "src/tests/test_data/time17_xy2_epsg3005.tif",
                [str(x) for x in np.arange(2010, 2027)],
            ),
            (
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg",
                "2014",
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

            resto_a = RestorationArea(
                restoration_polygon=resto_poly,
                restoration_start=resto_start,
                reference_years=ref_years,
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
            resto_start = "2021"
            ref_years = "2020"
            resto_poly = gpd.read_file(
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
            )

            with pytest.raises(
                ValueError,
            ):
                resto_a = RestorationArea(
                    restoration_polygon=resto_poly,
                    restoration_start=resto_start,
                    reference_polygons=resto_poly,
                    reference_years=ref_years,
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
            resto_start = "2021"
            resto_poly = gpd.read_file(
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
            )
            ref_years = "2020"

            with pytest.raises(
                ValueError,
            ):
                resto_a = RestorationArea(
                    restoration_polygon=resto_poly,
                    restoration_start=resto_start,
                    reference_polygons=resto_poly,
                    reference_years=ref_years,
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
            resto_start = "2021"
            resto_poly = gpd.read_file(
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
            )
            ref_years = "2020"

            with pytest.raises(
                ValueError,
            ):
                resto_a = RestorationArea(
                    restoration_polygon=resto_poly,
                    restoration_start=resto_start,
                    reference_polygons=resto_poly,
                    reference_years=ref_years,
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


class TestValidateDates:
    def test_only_dist_year_defaults_resto_year_to_next_year(self):
        dist_start = "2010"
        rest_start = None
        ref_years = "2009"
        test_stack = xr.DataArray(
            np.ones((1, 3, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"time": pd.date_range("2009", "2011", freq=DATETIME_FREQ)},
        )
        expected_rest = str(int(dist_start) + 1)

        ref, dist, rest = _validate_dates(
            disturbance_start=dist_start,
            restoration_start=rest_start,
            reference_years=ref_years,
            image_stack=test_stack,
        )

        assert rest == expected_rest

    def test_dist_year_greater_than_rest_year_throws_value_error(self):
        dist_start = "2011"
        rest_start = "2010"
        ref_years = "2009"
        test_stack = xr.DataArray(
            np.ones((1, 3, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"time": pd.date_range("2009", "2011", freq=DATETIME_FREQ)},
        )
        expected_rest = str(int(dist_start) + 1)

        with pytest.raises(ValueError):
            ref, dist, rest = _validate_dates(
                disturbance_start=dist_start,
                restoration_start=rest_start,
                reference_years=ref_years,
                image_stack=test_stack,
            )

    def test_only_rest_start_defaults_dist_year_to_prior_year(self):
        dist_start = None
        rest_start = "2011"
        ref_years = "2009"
        test_stack = xr.DataArray(
            np.ones((1, 3, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"time": pd.date_range("2009", "2011", freq=DATETIME_FREQ)},
        )
        expected_dist = str(int(rest_start) - 1)

        ref, dist, rest = _validate_dates(
            disturbance_start=dist_start,
            restoration_start=rest_start,
            reference_years=ref_years,
            image_stack=test_stack,
        )
        assert dist == expected_dist

    def test_out_of_bounds_restoration_start_year_throws_value_error(self):
        dist_start = "2011"
        rest_start = "2015"
        ref_years = "2009"
        test_stack = xr.DataArray(
            np.ones((1, 3, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"time": pd.date_range("2009", "2011", freq=DATETIME_FREQ)},
        )
        expected_rest = str(int(dist_start) + 1)

        with pytest.raises(ValueError):
            ref, dist, rest = _validate_dates(
                disturbance_start=dist_start,
                restoration_start=rest_start,
                reference_years=ref_years,
                image_stack=test_stack,
            )

    def test_out_of_bounds_disturbance_start_year_throws_value_error(self):
        dist_start = "2008"
        rest_start = "2015"
        ref_years = "2009"
        test_stack = xr.DataArray(
            np.ones((1, 3, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"time": pd.date_range("2009", "2011", freq=DATETIME_FREQ)},
        )
        expected_rest = str(int(dist_start) + 1)

        with pytest.raises(ValueError):
            ref, dist, rest = _validate_dates(
                disturbance_start=dist_start,
                restoration_start=rest_start,
                reference_years=ref_years,
                image_stack=test_stack,
            )

    @pytest.mark.parametrize(
        "ref_years",
        [
            "2002",
            ["2025", "2028"],
            ["2008", "2011"],
        ],
    )
    def test_out_of_bounds_reference_years_throw_value_error(self, ref_years):
        dist_start = "2008"
        rest_start = "2015"
        test_stack = xr.DataArray(
            np.ones((1, 3, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"time": pd.date_range("2009", "2011", freq=DATETIME_FREQ)},
        )
        expected_rest = str(int(dist_start) + 1)

        with pytest.raises(ValueError):
            ref, dist, rest = _validate_dates(
                disturbance_start=dist_start,
                restoration_start=rest_start,
                reference_years=ref_years,
                image_stack=test_stack,
            )


class TestRestorationAreaRecoveryTarget:
    @pytest.fixture()
    def valid_ra_build(self):
        # TODO: Simplify this to just use int coords and polygons that intersect. Shouldn't need to read the files.
        resto_poly = gpd.read_file("src/tests/test_data/polygon_inbound_epsg3005.gpkg")
        resto_start = "2015"
        ref_years = "2010"
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
            "restoration_start": resto_start,
            "reference_years": ref_years,
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
        valid_ra_build["reference_polygons"] = valid_ra_build["restoration_polygon"]
        median_pixel = MedianTarget(scale="pixel")

        with pytest.raises(TypeError):
            ra = RestorationArea(**valid_ra_build, recovery_target_method=median_pixel)


class TestRestorationAreaMetrics:
    restoration_start = "2015"
    reference_year = "2011"
    time_range = [str(x) for x in np.arange(2010, 2027)]
    baseline_array = xr.DataArray([[[1.0]], [[2.0]]])

    @pytest.fixture()
    def valid_resto_area(self):
        polygon = "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
        raster = "src/tests/test_data/time17_xy2_epsg3005.tif"

        with rioxarray.open_rasterio(raster, chunks="auto") as data:
            resto_poly = gpd.read_file(polygon)
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
                restoration_start=self.restoration_start,
                reference_polygons=resto_poly,
                reference_years=self.reference_year,
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
        expected_result = mocked_return.expand_dims(dim={"metric": [Metric.Y2R]})

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
        expected_result = mocked_return.expand_dims(dim={"metric": [Metric.YRYR]})

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
        expected_result = mocked_return.expand_dims(dim={"metric": [Metric.DNBR]})

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
        expected_result = mocked_return.expand_dims(dim={"metric": [Metric.RRI]})

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
        expected_result = mocked_return.expand_dims(dim={"metric": [Metric.R80P]})

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


class TestGetReferenceImageStack:
    # Note: ReferencSystem assumes dates are passed as datetime, not str
    @pytest.fixture()
    def test_stack_1(self):
        test_stack = xr.DataArray(
            np.ones((3, 3, 10, 10)) * 0,
            dims=["time", "band", "y", "x"],
            coords={
                "time": [0, 1, 2],
                "band": [0, 1, 2],
                "y": np.arange(0, 10),
                "x": np.arange(0, 10),
            },
        )
        return test_stack

    @pytest.fixture()
    def image_stack(self):
        test_raster = "src/tests/test_data/time3_xy2_epsg3005.tif"
        with rioxarray.open_rasterio(test_raster) as data:
            test_stack = data
            test_stack = test_stack.rename({"band": "time"})
            test_stack = test_stack.expand_dims(dim={"band": [0]})
            test_stack = test_stack.astype(
                np.float64
            )  # NOTE: if this conversion doesn't happen, test with count will fail. Filtered ints become 0.
            test_stack = test_stack.assign_coords(
                time=(pd.date_range("2007", "2009", freq=DATETIME_FREQ))
            )
        return test_stack

    def test_init_correctly_clips_with_single_polygon(self, image_stack):
        reference_polys = gpd.read_file(
            "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
        )
        expected_stack = xr.DataArray(
            [[[[[1.0]], [[2.0]], [[3.0]]]]], dims=["poly_id", "band", "time", "y", "x"]
        )

        rs = _get_reference_image_stack(
            reference_polygons=reference_polys,
            image_stack=image_stack,
        )

        assert npt.assert_array_equal(rs, expected_stack.data) is None
        assert expected_stack.dims == rs.dims

    def test_init_clips_with_multipolygons_correctly(self, image_stack):
        reference_polys = gpd.read_file(
            "src/tests/test_data/polygon_multi_inbound_epsg3005.gpkg"
        )
        # The multipolygon contains 2 polygons, which overlap with the lower-left and upper-right pixels.
        # Note: remember that arrays are flipped in the y-axis, so the lower-left pixel is at the top of the array.
        expected_stack = xr.DataArray(
            [
                [[
                    [[1.0, np.nan], [np.nan, np.nan]],
                    [[2.0, np.nan], [np.nan, np.nan]],
                    [[3.0, np.nan], [np.nan, np.nan]],
                ]],
                [[
                    [[np.nan, np.nan], [np.nan, 1.0]],
                    [[np.nan, np.nan], [np.nan, 2.0]],
                    [[np.nan, np.nan], [np.nan, 3.0]],
                ]],
            ],
            dims=["poly_id", "band", "time", "y", "x"],
        )

        rs = _get_reference_image_stack(
            reference_polygons=reference_polys,
            image_stack=image_stack,
        )

        assert npt.assert_array_equal(rs, expected_stack.data) is None
        assert expected_stack.dims == rs.dims
