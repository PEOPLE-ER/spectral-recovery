import pytest
import xarray as xr
import numpy as np
import rioxarray
import geopandas as gpd
import pandas as pd

from mock import patch
from unittest.mock import MagicMock
from geopandas.testing import assert_geodataframe_equal

from spectral_recovery.recovery_target import historic_average
from spectral_recovery.restoration import ReferenceSystem, RestorationArea
from spectral_recovery.enums import Metric

# TODO: move test data into their own folders, create temp dirs so individual tests
# don't conflict while reading the data
# https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data

DATETIME_FREQ = (  # TODO: should this be kept somewhere else in the project? Seem wrong that it's defined again here and in timeseries
    "YS"
)


class TestRestorationAreaInit:
    @pytest.mark.parametrize(
        ("resto_poly", "resto_year", "ref_years", "raster", "time_range"),
        [
            (
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg",
                pd.to_datetime("2011"),
                pd.to_datetime("2010"),
                "src/tests/test_data/time17_xy2_epsg3005.tif",
                [str(x) for x in np.arange(2010, 2027)],
            ),
            (
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg",
                pd.to_datetime("2014"),
                (pd.to_datetime("2010"), pd.to_datetime("2013")),
                "src/tests/test_data/time17_xy2_epsg3005.tif",
                [str(x) for x in np.arange(2010, 2027)],
            ),
        ],
    )
    def test_good_init(
        self,
        resto_poly,
        resto_year,
        ref_years,
        raster,
        time_range,
    ):
        with rioxarray.open_rasterio(raster, chunks="auto") as data:
            stack = data
            stack = stack.rename({"band": "time"})
            stack = stack.assign_coords(
                time=(pd.date_range(time_range[0], time_range[-1], freq=DATETIME_FREQ))
            )
            resto_poly = gpd.read_file("src/tests/test_data/polygon_inbound_epsg3005.gpkg")
            resto_a = RestorationArea(
                restoration_polygon=resto_poly,
                restoration_year=resto_year,
                reference_polygon=resto_poly,
                reference_years=ref_years,
                composite_stack=stack,
            )
            assert isinstance(resto_a.restoration_year, pd.Timestamp)
            assert (
                resto_a.restoration_polygon.geometry.geom_equals(resto_poly.geometry)
            ).all()
            assert isinstance(resto_a.reference_system, ReferenceSystem)
            assert resto_a.reference_system.reference_polygons.geom_equals(
                resto_poly.geometry
            ).all()
            assert resto_a.reference_system.reference_range == ref_years

    # check fro bad resto year, bad reference year, bad spatial location
    @pytest.mark.parametrize(
        ("resto_poly", "resto_year", "ref_years", "raster", "time_range"),
        [
            (
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg",
                pd.to_datetime("2028"),
                pd.to_datetime("2012"),
                "src/tests/test_data/time17_xy2_epsg3005.tif",
                [str(x) for x in np.arange(2010, 2027)],
            ),
            (
                "src/tests/test_data/polygon_inbound_epsg3005.gpkg",
                pd.to_datetime("2010"),
                pd.to_datetime("2009"),
                "src/tests/test_data/time17_xy2_epsg3005.tif",
                [str(x) for x in np.arange(2010, 2027)],
            ),
            (
                "src/tests/test_data/polygon_outbound_epsg3005.gpkg",
                pd.to_datetime("2015"),
                pd.to_datetime("2012"),
                "src/tests/test_data/time17_xy2_epsg3005.tif",
                [str(x) for x in np.arange(2010, 2027)],
            ),
            (
                "src/tests/test_data/polygon_overlap_epsg3005.gpkg",
                pd.to_datetime("2015"),
                pd.to_datetime("2012"),
                "src/tests/test_data/time17_xy2_epsg3005.tif",
                [str(x) for x in np.arange(2010, 2027)],
            ),
        ],
    )
    def test_not_contained_error(
        self,
        resto_poly,
        resto_year,
        ref_years,
        raster,
        time_range,
    ):
        with rioxarray.open_rasterio(raster, chunks="auto") as data:
            stack = data
            stack = stack.rename({"band": "time"})
            stack = stack.assign_coords(
                time=(pd.date_range(time_range[0], time_range[-1], freq=DATETIME_FREQ))
            )
            resto_poly = gpd.read_file(resto_poly)
            with pytest.raises(
                ValueError,
                match="Not contained! Better message soon!",
            ):
                resto_a = RestorationArea(
                    restoration_polygon=resto_poly,
                    restoration_year=resto_year,
                    reference_polygon=resto_poly,
                    reference_years=ref_years,
                    composite_stack=stack,
                )

# NOTE: SAME_XR is a hacky solution to get around "ValueErrors" that
# are thrown if you try to assert a mocked function was called with 
# more than one DataArray. The need for this sol. is likely a symptom of bad design
# in RestorationArea... but for now it stays to ensure correctness.
# Solution from: https://stackoverflow.com/questions/44640717
class SAME_XR:
    def __init__(self, xr: xr.DataArray):
        self.xr = xr

    def __eq__(self, other):
        return isinstance(other, xr.DataArray) and other.equals(self.xr)
    
    def __repr__(self): 
        return repr(self.xr)


class TestRestorationAreaMetrics:

    restoration_year = pd.to_datetime("2015")
    reference_year = pd.to_datetime("2012") 
    time_range = [str(x) for x in np.arange(2010, 2027)]
    baseline_array = xr.DataArray([[[1.0]],[[2.0]]])

    @pytest.fixture()
    def valid_resto_area(self):
        polygon = "src/tests/test_data/polygon_inbound_epsg3005.gpkg"
        raster = "src/tests/test_data/time17_xy2_epsg3005.tif"
        
        with rioxarray.open_rasterio(raster, chunks="auto") as data:
            resto_poly = gpd.read_file(polygon)
            stack = data
            stack = stack.rename({"band": "time"})
            stack = stack.assign_coords(
                time=(pd.date_range(self.time_range[0], self.time_range[-1], freq=DATETIME_FREQ))
            )
            stack = xr.concat([stack, stack], dim=pd.Index([0, 1], name="band"))
            resto_area = RestorationArea(
                restoration_polygon=resto_poly,
                restoration_year=self.restoration_year,
                reference_polygon=resto_poly,
                reference_years=self.reference_year,
                composite_stack=stack,
            )

            mock_target_return = {"recovery_target": self.baseline_array }
            resto_area.reference_system.recovery_target = MagicMock(return_value=mock_target_return)
        
        return resto_area

    # @pytest.mark.parametrize(
    #     ("metric_list", "metric_method"),
    #     [
    #         (
    #             [Metric.percent_recovered],
    #             "percent_recovered",
    #         ),
    #         (
    #             [Metric.Y2R],
    #             "Y2R",
    #         ),
    #         (
    #             [Metric.RI],
    #             "RI",
    #         ),
    #         (
    #             [Metric.dNBR],
    #             "dNBR",
    #         ),
    #     ],
    # )
    # @patch("spectral_recovery.restoration._stack_bands")
    # def test_switch_case(
    #     self, mock_stack, valid_resto_area, metric_list, metric_method, mocker
    # ):
    #     metric_patch = mocker.patch(f"spectral_recovery.restoration.{metric_method}")

    #     valid_resto_area.metrics(metric_list)

    #     assert metric_patch.call_count == 1
    #     assert mock_stack.call_count == 1

    # @patch("spectral_recovery.restoration._stack_bands")
    # def test_multiple_metrics(self, mock_stack, valid_resto_area, mocker):
    #     metrics_list = [
    #         Metric.percent_recovered,
    #         Metric.Y2R,
    #         Metric.RI,
    #         Metric.dNBR,
    #     ]
    #     methods_list = [
    #         "percent_recovered",
    #         "Y2R",
    #         "YrYr",
    #         "RI",
    #         "dNBR",
    #     ]

    #     patches = {}
    #     for i, metric in enumerate(metrics_list):
    #         patches[metric] = mocker.patch(
    #             f"spectral_recovery.restoration.{methods_list[i]}"
    #         )

    #     valid_resto_area.metrics(metrics_list)

    #     for i, metric in enumerate(metrics_list):
    #         assert patches[metric].call_count == 1
    #     assert mock_stack.call_count == 1

    @patch(
        "spectral_recovery.metrics.percent_recovered",
    )
    def test_percent_recovered_call_default(self, method_mock, valid_resto_area):
        mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
        method_mock.return_value = mocked_return

        result = valid_resto_area.percent_recovered()
        expected_result = mocked_return.expand_dims(dim={"metric": [Metric.percent_recovered]})

        assert result.equals(expected_result)

        final_obs = valid_resto_area.stack.sel(time=valid_resto_area.end_year)
        baseline =  xr.DataArray([[[1.0]],[[2.0]]])
        event_obs = valid_resto_area.stack.sel(time=valid_resto_area.restoration_year)

        method_mock.assert_called_with(
            eval_stack=SAME_XR(final_obs),
            recovery_target=SAME_XR(baseline),
            event_obs=SAME_XR(event_obs)
        )

    @patch(
        "spectral_recovery.metrics.Y2R",
    )
    def test_Y2R_call_default(self, method_mock, valid_resto_area):
        mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
        method_mock.return_value = mocked_return

        result = valid_resto_area.Y2R()
        expected_result = mocked_return.expand_dims(dim={"metric": [Metric.Y2R]})

        assert result.equals(expected_result)

        post_restoration = valid_resto_area.stack.sel(time=slice(valid_resto_area.restoration_year, None))
        rest_start = str(valid_resto_area.restoration_year.year)
        rest_end = str(valid_resto_area.end_year.year)
        default_percent = 80

        method_mock.assert_called_with(
            image_stack=SAME_XR(post_restoration),
            recovery_target=SAME_XR(self.baseline_array),
            rest_start=rest_start, 
            rest_end=rest_end,
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
            rest_start=str(valid_resto_area.restoration_year.year),
            timestep=timestep_default
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
            rest_start=str(valid_resto_area.restoration_year.year),
            timestep=timestep_default
        )

    @patch(
        "spectral_recovery.metrics.RI",
    )
    def test_RI_call_default(self, method_mock, valid_resto_area):
        mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
        method_mock.return_value = mocked_return

        result = valid_resto_area.RI()
        expected_result = mocked_return.expand_dims(dim={"metric": [Metric.RI]})

        assert result.equals(expected_result)

        timestep_default = 5

        method_mock.assert_called_with(
            image_stack=SAME_XR(valid_resto_area.stack),
            rest_start=str(valid_resto_area.restoration_year.year),
            timestep=timestep_default
        )
    
    @patch(
        "spectral_recovery.metrics.P80R",
    )
    def test_P80R_call_default(self, method_mock, valid_resto_area):
        mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
        method_mock.return_value = mocked_return

        result = valid_resto_area.P80R()
        expected_result = mocked_return.expand_dims(dim={"metric": [Metric.P80R]})

        assert result.equals(expected_result)

        percent_default = 80

        method_mock.assert_called_with(
            image_stack=SAME_XR(valid_resto_area.stack),
            rest_start=str(valid_resto_area.restoration_year.year),
            recovery_target=SAME_XR(self.baseline_array),
            percent=percent_default,
        )


    # @patch(
    #     "spectral_recovery.restoration.RestorationArea.percent_recovered",
    #     return_value=xr.DataArray([[1.0]], dims=["y", "x"]),
    # )
    # @patch(
    #     "spectral_recovery.restoration.RestorationArea.dNBR",
    #     return_value=xr.DataArray([[0.5]], dims=["y", "x"]),
    # )
    # def test_stack_multiple_metrics(
    #     self,
    #     percent_reco_return,
    #     dNBR_return,
    #     valid_resto_area,
    # ):
    #     metric = [Metric.percent_recovered, Metric.dNBR]
    #     expected_result = xr.DataArray(
    #         [[[1.0]], [[0.5]]],
    #         dims=["metric", "y", "x"],
    #         coords={"metric": [Metric.percent_recovered, Metric.dNBR]},
    #     )
    #     result = valid_resto_area.metrics(metric)
    #     assert result.equals(expected_result)

    # @patch(
    #     "spectral_recovery.restoration.RestorationArea.percent_recovered",
    #     return_value=xr.DataArray([[1.0]], dims=["y", "x"]),
    # )
    # def test_stack_single_metric(
    #     self,
    #     percent_reco_return,
    #     valid_resto_area,
    # ):
    #     metric = [Metric.percent_recovered]
    #     expected_result = xr.DataArray(
    #         [[[1.0]]],
    #         dims=["metric", "y", "x"],
    #         coords={"metric": [Metric.percent_recovered]},
    #     )
    #     result = valid_resto_area.metrics(metric)
    #     assert result.equals(expected_result)


class TestReferenceSystemInit:
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
            test_stack = test_stack.astype(
                np.float64
            )  # NOTE: if this conversion doesn't happen, test with count will fail. Filtered ints become 0.
            test_stack = test_stack.assign_coords(
                time=(pd.date_range("2007", "2009", freq=DATETIME_FREQ))
            )
        return test_stack

    def test_init(self, image_stack):
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
        assert rs.recovery_target_method == historic_average
        assert rs.reference_stack.count() == 3

    def test_init_multi_poly(self, image_stack):
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
        # assert_geodataframe_equal(rs.reference_polygons, reference_polys, check_geom_type=True)
        assert rs.reference_stack.count() == 6

    # NOTE: some of these test might be redundant? Might already be covered by testing of the Accessor contains methods?
    def test_not_contained_error_outbounds(self, image_stack):
        reference_polys = gpd.read_file(
            "src/tests/test_data/polygon_outbound_epsg3005.gpkg"
        )
        reference_date = pd.to_datetime("2008")

        with pytest.raises(
            ValueError,
            match="Not contained! Better message soon!",
        ):
            rs = ReferenceSystem(
                reference_polygons=reference_polys,
                reference_stack=image_stack,
                reference_range=reference_date,
                recovery_target_method=None,
            )

    def test_not_contained_error_overlap(self, image_stack):
        reference_poly_overlap = gpd.read_file(
            "src/tests/test_data/polygon_overlap_epsg3005.gpkg"
        )
        reference_date = pd.to_datetime("2008")

        with pytest.raises(
            ValueError,
            match="Not contained! Better message soon!",
        ):
            rs = ReferenceSystem(
                reference_polygons=reference_poly_overlap,
                reference_stack=image_stack,
                reference_range=reference_date,
                recovery_target_method=None,
            )

    def test_not_contained_error_multi_in_and_out(self, image_stack):
        reference_polys_multi = gpd.read_file(
            "src/tests/test_data/polygon_multi_inoutbound_epsg3005.gpkg"
        )
        reference_date = pd.to_datetime("2008")

        with pytest.raises(
            ValueError,
            match="Not contained! Better message soon!",
        ):
            rs = ReferenceSystem(
                reference_polygons=reference_polys_multi,
                reference_stack=image_stack,
                reference_range=reference_date,
                recovery_target_method=None,
            )

    def test_not_contained_error_time(self, image_stack):
        reference_polys = gpd.read_file(
            "src/tests/test_data/polygon_outbound_epsg3005.gpkg"
        )
        reference_date = pd.to_datetime("2020")

        with pytest.raises(
            ValueError,
            match="Not contained! Better message soon!",
        ):
            rs = ReferenceSystem(
                reference_polygons=reference_polys,
                reference_stack=image_stack,
                reference_range=reference_date,
                recovery_target_method=None,
            )


class TestReferenceSystemrecovery_target:

    class SimpleReferenceSystem(ReferenceSystem):
        """Sub-class ReferenceSystem and overwrite __init__ to isolate `recovery_target` method."""

        def __init__(self, recovery_target, stack, date):
            """Set only attributes that are required by `recovery_target`, assume arb. types"""
            self.recovery_target_method = recovery_target
            self.reference_stack = stack
            self.reference_range = date

    def test_recovery_target_is_called_with_args(self):
        mock_value = 3.0
        mock_recovery_target = MagicMock(return_value=mock_value)
        rs = self.SimpleReferenceSystem(mock_recovery_target, 1.0, 2.0)
        output = rs.recovery_target()
        expected = {"recovery_target": mock_value}
        assert output == expected
