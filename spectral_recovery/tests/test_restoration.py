import pytest
from spectral_recovery.baselines import historic_average
import xarray as xr
import numpy as np
import rioxarray
import geopandas as gpd
import pandas as pd

from mock import patch
from unittest.mock import MagicMock
from geopandas.testing import assert_geodataframe_equal
from spectral_recovery.restoration import ReferenceSystem, RestorationArea
from spectral_recovery.enums import Metric

# TODO: move test data into their own folders, create temp dirs so individual tests
# don't conflict while reading the data
# https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data

DATETIME_FREQ = "YS"  # TODO: should this be kept somewhere else in the project? Seem wrong that it's defined again here and in timeseries


class TestRestorationAreaInit:
    # This might need some set-up and tear down
    # https://stackoverflow.com/questions/26405380/how-do-i-correctly-setup-and-teardown-for-my-pytest-class-with-tests
    # TODO: test that RA contains the clipped data for a polygon, not the entire AOI
    @pytest.mark.parametrize(
        ("resto_poly", "resto_year", "ref_sys", "raster", "end_year", "time_range"),
        [
            (
                "1p_test.gpkg",
                pd.to_datetime("2011"),
                pd.to_datetime("2010"),
                "test_recovered.tif",
                None,
                [str(x) for x in np.arange(2010, 2022)],
            ),
            (
                "1p_test.gpkg",
                pd.to_datetime("2014"),
                (pd.to_datetime("2010"), pd.to_datetime("2013")),
                "test_recovered.tif",
                None,
                [str(x) for x in np.arange(2010, 2022)],
            ),
        ],
    )
    def test_good_init(
        self,
        resto_poly,
        resto_year,
        ref_sys,
        raster,
        end_year,
        time_range,
    ):
        with rioxarray.open_rasterio(raster, chunks="auto") as data:
            stack = data
            stack = stack.rename({"band": "time"})
            stack = stack.assign_coords(
                time=(pd.date_range(time_range[0], time_range[-1], freq=DATETIME_FREQ))
            )
            resto_poly = gpd.read_file("1p_test.gpkg")
            resto_a = RestorationArea(
                restoration_polygon=resto_poly,
                restoration_year=resto_year,
                reference_system=ref_sys,
                composite_stack=stack,
                end_year=end_year,
            )
            assert isinstance(resto_a.restoration_year, pd.Timestamp)
            assert (
                resto_a.restoration_polygon.geometry.geom_equals(resto_poly.geometry)
            ).all()
            assert isinstance(resto_a.reference_system, ReferenceSystem)
            assert resto_a.reference_system.reference_polygons.geom_equals(
                resto_poly.geometry
            ).all()
            assert resto_a.reference_system.reference_range == ref_sys

    # check fro bad resto year, bad reference year, bad spatial location
    @pytest.mark.parametrize(
        ("resto_poly", "resto_year", "ref_sys", "raster", "end_year", "time_range"),
        [
            (
                "1p_test.gpkg",
                pd.to_datetime("2023"),
                pd.to_datetime("2012"),
                "test_recovered.tif",
                None,
                [str(x) for x in np.arange(2010, 2022)],
            ),
            (
                "1p_test.gpkg",
                pd.to_datetime("2010"),
                pd.to_datetime("2009"),
                "test_recovered.tif",
                None,
                [str(x) for x in np.arange(2010, 2022)],
            ),
            (
                "no_overlap.gpkg",
                pd.to_datetime("2015"),
                pd.to_datetime("2012"),
                "test_recovered.tif",
                None,
                [str(x) for x in np.arange(2010, 2022)],
            ),
            (
                "not_fully_contained.gpkg",
                pd.to_datetime("2015"),
                pd.to_datetime("2012"),
                "test_recovered.tif",
                None,
                [str(x) for x in np.arange(2010, 2022)],
            ),
        ],
    )
    def test_not_contained_error(
        self,
        resto_poly,
        resto_year,
        ref_sys,
        raster,
        end_year,
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
                    reference_system=ref_sys,
                    composite_stack=stack,
                    end_year=end_year,
                )


class TestRestorationAreaMetrics:
    @pytest.fixture()
    def valid_resto_area(self):
        polygon = "1p_test.gpkg"
        restoration_year = pd.to_datetime("2015")
        reference_year = pd.to_datetime("2012")
        raster = "test_recovered.tif"
        time_range = [str(x) for x in np.arange(2010, 2022)]

        with rioxarray.open_rasterio(raster, chunks="auto") as data:
            resto_poly = gpd.read_file(polygon)
            stack = data
            stack = stack.rename({"band": "time"})
            stack = stack.assign_coords(
                time=(pd.date_range(time_range[0], time_range[-1], freq=DATETIME_FREQ))
            )
            resto_area = RestorationArea(
                restoration_polygon=resto_poly,
                restoration_year=restoration_year,
                reference_system=reference_year,
                composite_stack=stack,
                end_year=None,
            )
        return resto_area

    @pytest.mark.parametrize(
        ("metric_list", "metric_method"),
        [
            (
                [Metric.percent_recovered],
                "percent_recovered",
            ),
            (
                [Metric.years_to_recovery],
                "years_to_recovery",
            ),
            (
                [Metric.recovery_indicator],
                "recovery_indicator",
            ),
            (
                [Metric.dNBR],
                "dNBR",
            ),
        ],
    )
    @patch("spectral_recovery.restoration._stack_bands")
    def test_switch_case(
        self, mock_stack, valid_resto_area, metric_list, metric_method, mocker
    ):
        metric_patch = mocker.patch(f"spectral_recovery.restoration.{metric_method}")

        valid_resto_area.metrics(metric_list)

        assert metric_patch.call_count == 1
        assert mock_stack.call_count == 1

    @patch("spectral_recovery.restoration._stack_bands")
    def test_multiple_metrics(self, mock_stack, valid_resto_area, mocker):
        metrics_list = [
            Metric.percent_recovered,
            Metric.years_to_recovery,
            Metric.recovery_indicator,
            Metric.dNBR,
        ]
        methods_list = [
            "percent_recovered",
            "years_to_recovery",
            "recovery_indicator",
            "dNBR",
        ]

        patches = {}
        for i, metric in enumerate(metrics_list):
            patches[metric] = mocker.patch(
                f"spectral_recovery.restoration.{methods_list[i]}"
            )

        valid_resto_area.metrics(metrics_list)

        for i, metric in enumerate(metrics_list):
            assert patches[metric].call_count == 1
        assert mock_stack.call_count == 1

    @patch(
        "spectral_recovery.restoration.percent_recovered",
        return_value=xr.DataArray([[1.0]], dims=["y", "x"]),
    )
    @patch(
        "spectral_recovery.restoration.dNBR",
        return_value=xr.DataArray([[0.5]], dims=["y", "x"]),
    )
    def test_stack_multiple_metrics(
        self,
        percent_reco_return,
        dNBR_return,
        valid_resto_area,
    ):
        metric = [Metric.percent_recovered, Metric.dNBR]
        expected_result = xr.DataArray(
            [[[1.0]], [[0.5]]],
            dims=["metric", "y", "x"],
            coords={"metric": [Metric.percent_recovered, Metric.dNBR]},
        )
        result = valid_resto_area.metrics(metric)
        assert result.equals(expected_result)

    @patch(
        "spectral_recovery.restoration.percent_recovered",
        return_value=xr.DataArray([[1.0]], dims=["y", "x"]),
    )
    def test_stack_single_metric(
        self,
        percent_reco_return,
        valid_resto_area,
    ):
        metric = [Metric.percent_recovered]
        expected_result = xr.DataArray(
            [[[1.0]]],
            dims=["metric", "y", "x"],
            coords={"metric": [Metric.percent_recovered]},
        )
        result = valid_resto_area.metrics(metric)
        assert result.equals(expected_result)


class TestReferenceSystemInit:
    # @pytest.fixture()
    # def gdf(self):
    #     p1 = Polygon([(0, 0), (1, 0), (1, 1)])
    #     p2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    #     g = gpd.GeoSeries([p1, p2])
    #     gdf = gpd.GeoDataFrame(geometry=g)
    #     return gdf
    # @pytest.fixture()
    # def reference_date(self):
    #     return pd.to_datetime("2008")

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
        test_raster = "spectral_recovery/tests/test_data/time3_xy2_epsg3005.tif"
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
            "spectral_recovery/tests/test_data/polygon_inbound_epsg3005.gpkg"
        )
        reference_date = pd.to_datetime("2008")

        rs = ReferenceSystem(
            reference_polygons=reference_polys,
            reference_stack=image_stack,
            reference_range=reference_date,
            baseline_method=None,
        )
        assert_geodataframe_equal(
            rs.reference_polygons, reference_polys, check_geom_type=True
        )
        assert rs.reference_range == reference_date
        assert rs.baseline_method == historic_average
        assert rs.reference_stack.count() == 3

    def test_init_multi_poly(self, image_stack):
        reference_polys = gpd.read_file(
            "spectral_recovery/tests/test_data/polygon_multi_inbound_epsg3005.gpkg"
        )
        reference_date = pd.to_datetime("2008")

        rs = ReferenceSystem(
            reference_polygons=reference_polys,
            reference_stack=image_stack,
            reference_range=reference_date,
            baseline_method=None,
        )
        # assert_geodataframe_equal(rs.reference_polygons, reference_polys, check_geom_type=True)
        assert rs.reference_stack.count() == 6

    # NOTE: some of these test might be redundant? Might already be covered by testing of the Accessor contains methods?
    def test_not_contained_error_outbounds(self, image_stack):
        reference_polys = gpd.read_file(
            "spectral_recovery/tests/test_data/polygon_outbound_epsg3005.gpkg"
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
                baseline_method=None,
            )

    def test_not_contained_error_overlap(self, image_stack):
        reference_poly_overlap = gpd.read_file(
            "spectral_recovery/tests/test_data/polygon_overlap_epsg3005.gpkg"
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
                baseline_method=None,
            )

    def test_not_contained_error_multi_in_and_out(self, image_stack):
        reference_polys_multi = gpd.read_file(
            "spectral_recovery/tests/test_data/polygon_multi_inoutbound_epsg3005.gpkg"
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
                baseline_method=None,
            )

    def test_not_contained_error_time(self, image_stack):
        reference_polys = gpd.read_file(
            "spectral_recovery/tests/test_data/polygon_outbound_epsg3005.gpkg"
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
                baseline_method=None,
            )


class TestReferenceSystemBaseline:
    class TestingReferenceSystem(ReferenceSystem):
        """Sub-class ReferenceSystem and overwrite __init__ to isolate `baseline` method."""

        def __init__(self, baseline, stack, date):
            """Set only attributes that are required by `baseline`, assume arb. types"""
            self.baseline_method = baseline
            self.reference_stack = stack
            self.reference_range = date

    def test_baseline_is_called_with_args(self):
        mock_value = 3.0
        mock_baseline = MagicMock(return_value=mock_value)
        rs = self.TestingReferenceSystem(mock_baseline, 1.0, 2.0)
        output = rs.baseline()
        expected = {"baseline": mock_value}
        assert output == expected
