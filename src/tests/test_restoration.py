import pytest
import xarray as xr
import numpy as np
import rioxarray
import geopandas as gpd
import pandas as pd

from unittest.mock import patch
from unittest.mock import MagicMock
from numpy import testing as npt
from geopandas.testing import assert_geodataframe_equal
from tests.utils import SAME_XR

from spectral_recovery.recovery_target import median_target
from spectral_recovery.restoration import ReferenceSystem, RestorationArea
from spectral_recovery.enums import Metric
from  spectral_recovery.config import DATETIME_FREQ

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
                ["2010", "2013"],
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
                reference_polygon=resto_poly,
                reference_years=ref_years,
                composite_stack=stack,
            )

            resto_start_dt = pd.to_datetime(resto_start)
            ref_years_dt = pd.to_datetime(ref_years)

            assert isinstance(resto_a.restoration_start, pd.Timestamp)
            assert (
                resto_a.restoration_polygon.geometry.geom_equals(resto_poly.geometry)
            ).all()
            assert resto_a.restoration_start == resto_start_dt
            assert isinstance(resto_a.reference_system, ReferenceSystem)
            assert resto_a.reference_system.reference_polygons.geom_equals(
                resto_poly.geometry
            ).all()
            if isinstance(ref_years, list):
                assert (resto_a.reference_system.reference_range == ref_years_dt).all()
            else:
                assert resto_a.reference_system.reference_range == ref_years_dt

    
    def test_only_dist_year_defaults_resto_year_to_next_year(self):
        resto_poly = gpd.read_file("src/tests/test_data/polygon_inbound_epsg3005.gpkg")
        dist_start = "2015"
        ref_years = "2010"
        raster = "src/tests/test_data/time17_xy2_epsg3005.tif"
        time_range = [str(x) for x in np.arange(2010, 2027)]

        expected_resto_start = pd.to_datetime("2016")  # + 1 year from dist_start, in datetime form

        with rioxarray.open_rasterio(raster, chunks="auto") as data:
            stack = data
            stack = stack.rename({"band": "time"})
            stack = stack.expand_dims(dim={"band": [0]})
            stack = stack.assign_coords(
                time=(pd.date_range(time_range[0], time_range[-1], freq=DATETIME_FREQ))
            )
            print(stack)
            resto_a = RestorationArea(
                restoration_polygon=resto_poly,
                disturbance_start=dist_start,
                reference_polygon=resto_poly,
                reference_years=ref_years,
                composite_stack=stack,
            )
            assert resto_a.restoration_start == expected_resto_start

    def test_dist_year_greater_than_rest_year_throws_value_error(self):
        resto_poly = gpd.read_file("src/tests/test_data/polygon_inbound_epsg3005.gpkg")
        resto_start = "2015"
        dist_start = "2016"
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
            with pytest.raises(
                ValueError,
                match=(
                    "The disturbance start year must be less than the restoration start"
                    " year."
                ),
            ):
                resto_a = RestorationArea(
                    restoration_polygon=resto_poly,
                    restoration_start=resto_start,
                    disturbance_start=dist_start,
                    reference_polygon=resto_poly,
                    reference_years=ref_years,
                    composite_stack=stack,
                )

    def test_only_rest_start_defaults_dist_year_to_prior_year(self):
        resto_poly = gpd.read_file("src/tests/test_data/polygon_inbound_epsg3005.gpkg")
        resto_start = "2015"
        ref_years = "2010"
        raster = "src/tests/test_data/time17_xy2_epsg3005.tif"
        time_range = [str(x) for x in np.arange(2010, 2027)]

        expected_dist_start_dt = pd.to_datetime("2014")
        with rioxarray.open_rasterio(raster, chunks="auto") as data:
            stack = data
            stack = stack.rename({"band": "time"})
            stack = stack.expand_dims(dim={"band": [0]})
            stack = stack.assign_coords(
                time=(pd.date_range(time_range[0], time_range[-1], freq=DATETIME_FREQ))
            )
            resto_a = RestorationArea(
                restoration_polygon=resto_poly,
                restoration_start=resto_start,
                reference_polygon=resto_poly,
                reference_years=ref_years,
                composite_stack=stack,
            )
            assert resto_a.disturbance_start == expected_dist_start_dt

    def test_out_of_bounds_restoration_start_year_throws_value_error(self):
        with rioxarray.open_rasterio(
            "src/tests/test_data/time17_xy2_epsg3005.tif", chunks="auto"
        ) as data:
            stack = data
            stack = stack.rename({"band": "time"})
            stack = stack.expand_dims(dim={"band": [0]})
            stack = stack.assign_coords(
                time=(pd.date_range("2010", "2026", freq=DATETIME_FREQ))
            )
            resto_poly = gpd.read_file(
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
            )
            # stack's temporal range is 2010-2026, set resto_start to greater than 2026
            ref_years = "2010"
            dist_start = "2011"
            resto_start = "2028"  # bad value!

            with pytest.raises(
                ValueError,
            ):
                resto_a = RestorationArea(
                    restoration_polygon=resto_poly,
                    restoration_start=resto_start,
                    disturbance_start=dist_start,
                    reference_polygon=resto_poly,
                    reference_years=ref_years,
                    composite_stack=stack,
                )

    def test_out_of_bounds_disturbance_start_year_throws_value_error(self):
        with rioxarray.open_rasterio(
            "src/tests/test_data/time17_xy2_epsg3005.tif", chunks="auto"
        ) as data:
            stack = data
            stack = stack.rename({"band": "time"})
            stack = stack.expand_dims(dim={"band": [0]})
            stack = stack.assign_coords(
                time=(pd.date_range("2010", "2026", freq=DATETIME_FREQ))
            )
            resto_poly = gpd.read_file(
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
            )
            # stack's temporal range is 2010-2026, set dist_start to less than 2010
            ref_years = "2010"
            dist_start = "2005"
            resto_start = "2012"

            with pytest.raises(
                ValueError,
            ):
                resto_a = RestorationArea(
                    restoration_polygon=resto_poly,
                    restoration_start=resto_start,
                    disturbance_start=dist_start,
                    reference_polygon=resto_poly,
                    reference_years=ref_years,
                    composite_stack=stack,
                )

    @pytest.mark.parametrize(
        "ref_years",
        [
            ["2002"],
            ["2025", "2028"],
            ["2008","2012"],
        ],
    )
    def test_out_of_bounds_reference_years_throw_value_error(self, ref_years):
        with rioxarray.open_rasterio(
            "src/tests/test_data/time17_xy2_epsg3005.tif", chunks="auto"
        ) as data:
            stack = data
            stack = stack.rename({"band": "time"})
            stack = stack.expand_dims(dim={"band": [0]})
            stack = stack.assign_coords(
                time=(pd.date_range("2010", "2026", freq=DATETIME_FREQ))
            )
            resto_poly = gpd.read_file(
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
            )
            # stack's temporal range is 2010-2025
            # reference years taken from pytest.parametrize
            dist_start = "2013"
            resto_start = "2014"

            with pytest.raises(
                ValueError,
            ):
                resto_a = RestorationArea(
                    restoration_polygon=resto_poly,
                    restoration_start=resto_start,
                    disturbance_start=dist_start,
                    reference_polygon=resto_poly,
                    reference_years=ref_years,
                    composite_stack=stack,
                )

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
            (  # bad spatial location (not contained at all)
                "src/tests/test_data/polygon_outbound_epsg3005.gpkg",
                "2015",
                None,
                "2012",
                "src/tests/test_data/time17_xy2_epsg3005.tif",
                [str(x) for x in np.arange(2010, 2027)],
            ),
            (  # bad spatial location (not fully contained)
                "src/tests/test_data/polygon_overlap_epsg3005.gpkg",
                "2015",
                None,
                "2012",
                "src/tests/test_data/time17_xy2_epsg3005.tif",
                [str(x) for x in np.arange(2010, 2027)],
            ),
        ],
    )
    def test_out_of_bounds_polygons_throw_value_err(
        self, resto_poly, resto_start, dist_start, ref_years, raster, time_range
    ):
        with rioxarray.open_rasterio(raster, chunks="auto") as data:
            stack = data
            stack = stack.rename({"band": "time"})
            stack = stack.expand_dims(dim={"band": [0]})
            stack = stack.assign_coords(
                time=(pd.date_range(time_range[0], time_range[-1], freq=DATETIME_FREQ))
            )
            resto_poly = gpd.read_file(resto_poly)
            with pytest.raises(
                ValueError,
            ):
                resto_a = RestorationArea(
                    restoration_polygon=resto_poly,
                    restoration_start=resto_start,
                    disturbance_start=dist_start,
                    reference_polygon=resto_poly,
                    reference_years=ref_years,
                    composite_stack=stack,
                )

    def test_composite_stack_wrong_dims_throws_value_error(self):
        with rioxarray.open_rasterio(
            "src/tests/test_data/time3_xy2_epsg3005.tif"
        ) as data:
            # Misname band dimension to "bandz"
            bad_stack = data
            bad_stack = bad_stack.rename({"band": "time"})
            bad_stack = bad_stack.expand_dims(dim={"bandz": [0]})
            bad_stack = bad_stack.assign_coords(
                time=(
                    [
                        pd.to_datetime("2020"),
                        pd.to_datetime("2021"),
                        pd.to_datetime("2022"),
                    ]
                )
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
                    reference_polygon=resto_poly,
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
                time=(
                    [
                        pd.to_datetime("2020"),
                        pd.to_datetime("2021"),
                        pd.to_datetime("2023"),
                    ]
                )
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
                    reference_polygon=resto_poly,
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
                time=(
                    [
                        pd.to_datetime("2020"),
                        pd.to_datetime("2021"),
                        pd.to_datetime("2023"),
                    ]
                )
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
                    reference_polygon=resto_poly,
                    reference_years=ref_years,
                    composite_stack=bad_stack,
                )


class TestRestorationAreaMetrics:
    restoration_start = "2015"
    reference_year = "2012"
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
                reference_polygon=resto_poly,
                reference_years=self.reference_year,
                composite_stack=stack,
            )

            mock_target_return = self.baseline_array
            resto_area.reference_system.recovery_target = MagicMock(
                return_value=mock_target_return
            )

        return resto_area

    @patch(
        "spectral_recovery.metrics.Y2R",
    )
    def test_Y2R_call_default(self, method_mock, valid_resto_area):
        mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
        method_mock.return_value = mocked_return

        result = valid_resto_area.Y2R()
        expected_result = mocked_return.expand_dims(dim={"metric": [Metric.Y2R]})

        assert result.equals(expected_result)

        post_restoration = valid_resto_area.stack.sel(
            time=slice(valid_resto_area.restoration_start, None)
        )
        rest_start = str(valid_resto_area.restoration_start.year)
        rest_end = str(valid_resto_area.end_year.year)
        default_percent = 80

        method_mock.assert_called_with(
            image_stack=SAME_XR(post_restoration),
            recovery_target=SAME_XR(self.baseline_array),
            rest_start=rest_start,
            percent=default_percent,
        )

    @patch(
        "spectral_recovery.metrics.YrYr",
    )
    def test_YrYr_call_default(self, method_mock, valid_resto_area):
        mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
        method_mock.return_value = mocked_return

        result = valid_resto_area.YrYr()
        expected_result = mocked_return.expand_dims(dim={"metric": [Metric.YrYr]})

        assert result.equals(expected_result)

        timestep_default = 5

        method_mock.assert_called_with(
            image_stack=SAME_XR(valid_resto_area.stack),
            rest_start=str(valid_resto_area.restoration_start.year),
            timestep=timestep_default,
        )

    @patch(
        "spectral_recovery.metrics.dNBR",
    )
    def test_dNBR_call_default(self, method_mock, valid_resto_area):
        mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
        method_mock.return_value = mocked_return

        result = valid_resto_area.dNBR()
        expected_result = mocked_return.expand_dims(dim={"metric": [Metric.dNBR]})

        assert result.equals(expected_result)

        timestep_default = 5

        method_mock.assert_called_with(
            image_stack=SAME_XR(valid_resto_area.stack),
            rest_start=str(valid_resto_area.restoration_start.year),
            timestep=timestep_default,
        )

    @patch(
        "spectral_recovery.metrics.RRI",
    )
    def test_RRI_call_default(self, method_mock, valid_resto_area):
        mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
        method_mock.return_value = mocked_return

        result = valid_resto_area._RRI()
        expected_result = mocked_return.expand_dims(dim={"metric": [Metric.RRI]})

        assert result.equals(expected_result)

        timestep_default = 5

        method_mock.assert_called_with(
            image_stack=SAME_XR(valid_resto_area.stack),
            rest_start=str(valid_resto_area.restoration_start.year),
            dist_start=str(valid_resto_area.disturbance_start.year),
            timestep=timestep_default,
        )

    @patch(
        "spectral_recovery.metrics.R80P",
    )
    def test_R80P_call_default(self, method_mock, valid_resto_area):
        mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
        method_mock.return_value = mocked_return

        result = valid_resto_area.R80P()
        expected_result = mocked_return.expand_dims(dim={"metric": [Metric.R80P]})

        assert result.equals(expected_result)

        percent_default = 80
        timestep_default = 5

        method_mock.assert_called_with(
            image_stack=SAME_XR(valid_resto_area.stack),
            rest_start=str(valid_resto_area.restoration_start.year),
            recovery_target=SAME_XR(self.baseline_array),
            timestep=timestep_default,
            percent=percent_default,
        )


class TestReferenceSystemInit:
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

    def test_init_correctly_sets_dates(self, image_stack):
        reference_polys = gpd.read_file(
            "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
        )
        reference_date = pd.to_datetime("2008")

        rs = ReferenceSystem(
            reference_polygons=reference_polys,
            reference_stack=image_stack,
            reference_range=reference_date,
            recovery_target_method=None,
        )
        assert_geodataframe_equal(
            rs.reference_polygons, reference_polys, check_geom_type=True
        )
        assert rs.reference_range == reference_date
        assert rs.recovery_target_method == median_target
        # the polygon overlaps only with the lower-right pixel of the stack.

    def test_init_correctly_clips_with_single_polygon(self, image_stack):
        reference_polys = gpd.read_file(
            "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
        )
        reference_date = pd.to_datetime("2008")

        rs = ReferenceSystem(
            reference_polygons=reference_polys,
            reference_stack=image_stack,
            reference_range=reference_date,
            recovery_target_method=None,
        )
        # create 4d data array with dims poly_id, band, time, y, x
        expected_stack = xr.DataArray(
            [[[[[1.0]], [[2.0]], [[3.0]]]]], dims=["poly_id", "band", "time", "y", "x"]
        )
        assert (
            npt.assert_array_equal(rs.reference_stack.data, expected_stack.data) is None
        )
        assert expected_stack.dims == rs.reference_stack.dims

    def test_init_clips_with_multipolygons_correctly(self, image_stack):
        reference_polys = gpd.read_file(
            "src/tests/test_data/polygon_multi_inbound_epsg3005.gpkg"
        )
        reference_date = pd.to_datetime("2008")

        rs = ReferenceSystem(
            reference_polygons=reference_polys,
            reference_stack=image_stack,
            reference_range=reference_date,
            recovery_target_method=None,
        )
        # The multipolygon contains 2 polygons, which overlap with the lower-left and upper-right pixels.
        # Note: remember that arrays are flipped in the y-axis, so the lower-left pixel is at the top of the array.
        expected_stack = xr.DataArray(
            [
                [
                    [
                        [[1.0, np.nan], [np.nan, np.nan]],
                        [[2.0, np.nan], [np.nan, np.nan]],
                        [[3.0, np.nan], [np.nan, np.nan]],
                    ]
                ],
                [
                    [
                        [[np.nan, np.nan], [np.nan, 1.0]],
                        [[np.nan, np.nan], [np.nan, 2.0]],
                        [[np.nan, np.nan], [np.nan, 3.0]],
                    ]
                ],
            ],
            dims=["poly_id", "band", "time", "y", "x"],
        )
        assert (
            npt.assert_array_equal(rs.reference_stack.data, expected_stack.data) is None
        )
        assert expected_stack.dims == rs.reference_stack.dims

    # NOTE: some of these test might be redundant? Might already be covered by testing of the Accessor contains methods?
    def test_out_of_bounds_polygon_throws_value_err(self, image_stack):
        reference_polys = gpd.read_file(
            "src/tests/test_data/polygon_outbound_epsg3005.gpkg"
        )
        reference_date = pd.to_datetime("2008")

        with pytest.raises(
            ValueError,
        ):
            rs = ReferenceSystem(
                reference_polygons=reference_polys,
                reference_stack=image_stack,
                reference_range=reference_date,
                recovery_target_method=None,
            )

    def test_overlapping_polygon_throws_value_err(self, image_stack):
        reference_poly_overlap = gpd.read_file(
            "src/tests/test_data/polygon_overlap_epsg3005.gpkg"
        )
        reference_date = pd.to_datetime("2008")

        with pytest.raises(
            ValueError,
        ):
            rs = ReferenceSystem(
                reference_polygons=reference_poly_overlap,
                reference_stack=image_stack,
                reference_range=reference_date,
                recovery_target_method=None,
            )

    def test_multipolygon_with_some_polygons_out_of_bounds_throws_value_err(
        self, image_stack
    ):
        reference_polys_multi = gpd.read_file(
            "src/tests/test_data/polygon_multi_inoutbound_epsg3005.gpkg"
        )
        reference_date = pd.to_datetime("2008")

        with pytest.raises(
            ValueError,
        ):
            rs = ReferenceSystem(
                reference_polygons=reference_polys_multi,
                reference_stack=image_stack,
                reference_range=reference_date,
                recovery_target_method=None,
            )

    def test_out_of_bounds_date_throws_value_err(self, image_stack):
        reference_polys = gpd.read_file(
            "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
        )
        reference_date = pd.to_datetime("2020")

        with pytest.raises(
            ValueError,
        ):
            rs = ReferenceSystem(
                reference_polygons=reference_polys,
                reference_stack=image_stack,
                reference_range=reference_date,
                recovery_target_method=None,
            )
    
    def test_historic_reference_system_bool_default_is_False(self, image_stack):
        reference_polys = gpd.read_file(
            "src/tests/test_data/polygon_multi_inbound_epsg3005.gpkg"
        )
        reference_date = pd.to_datetime("2008")

        rs = ReferenceSystem(
            reference_polygons=reference_polys,
            reference_stack=image_stack,
            reference_range=reference_date,
            recovery_target_method=None,
        )
        assert rs.hist_ref_sys == False
    
    def test_historic_reference_system_bool_is_set_True(self, image_stack):
        reference_polys = gpd.read_file(
            "src/tests/test_data/polygon_multi_inbound_epsg3005.gpkg"
        )
        reference_date = pd.to_datetime("2008")

        rs = ReferenceSystem(
            reference_polygons=reference_polys,
            reference_stack=image_stack,
            reference_range=reference_date,
            recovery_target_method=None,
            historic_reference_system=True,
        )
        assert rs.hist_ref_sys == True


class TestReferenceSystemRecoveryTarget:

    def test_false_hist_ref_sys_calls_recovery_target_with_space_true(self, mocker):
        mocker.patch.object(ReferenceSystem, "__init__", return_value=None)
        rs = ReferenceSystem()
        rs.recovery_target_method = MagicMock(return_value=None)
        rs.reference_stack = 0
        rs.reference_range = 0
        rs.hist_ref_sys = False

        rs.recovery_target()
        rs.recovery_target_method.assert_called_once()
        rs.recovery_target_method.assert_called_with(reference_stack=0, reference_range=0, space=True)
    
    def test_hist_ref_sys_calls_recovery_target_with_space_false(self, mocker):
        mocker.patch.object(ReferenceSystem, "__init__", return_value=None)
        rs = ReferenceSystem()
        rs.recovery_target_method = MagicMock(return_value=None)
        rs.reference_stack = 0
        rs.reference_range = 0
        rs.hist_ref_sys = True

        rs.recovery_target()
        rs.recovery_target_method.assert_called_once()
        rs.recovery_target_method.assert_called_with(reference_stack=0, reference_range=0, space=False)
    
    # TODO: test the return value is correct
