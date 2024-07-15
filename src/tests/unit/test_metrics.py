import pytest
import xarray as xr
import numpy as np
import pandas as pd
import rioxarray
import geopandas as gpd

from unittest.mock import patch, Mock
from shapely import Polygon

# from spectral_recovery.restoration import RestorationArea
from spectral_recovery.metrics import (
    y2r,
    dnbr,
    rri,
    yryr,
    r80p,
    METRIC_FUNCS,
    compute_metrics,
)


def test_metric_funcs_global_contains_all_funcs():
    expected_dict = {"y2r": y2r, "dnbr": dnbr, "yryr": yryr, "r80p": r80p, "rri": rri}
    assert METRIC_FUNCS == expected_dict


class TestComputeMetrics:

    valid_poly = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

    @pytest.fixture()
    def valid_array(self):
        data = np.ones((2, 5, 2, 2))
        latitudes = [0, 1]
        longitudes = [0, 1]
        time = pd.date_range("2010", "2014", freq="YS")
        xarr = xr.DataArray(
            data,
            dims=["band", "time", "y", "x"],
            coords={"band": ["N", "R"], "time": time, "y": latitudes, "x": longitudes},
        )
        xarr.rio.write_crs("EPSG:4326", inplace=True)
        return xarr

    @pytest.fixture()
    def valid_frame(self):

        valid_frame = gpd.GeoDataFrame(
            {
                "dist_start": [2012],
                "rest_start": [2013],
                "geometry": [self.valid_poly],
            },
            crs="EPSG:4326",
        )
        return valid_frame

    @pytest.fixture()
    def valid_rt(self, valid_array):

        valid_rt = valid_array[:,0, :, :].drop_vars("time")
        return valid_rt


    def test_none_rt_passes(
        self, valid_array, valid_frame
    ):

        y2r_mock = Mock()
        y2r_mock.return_value = xr.DataArray([[[0.0]]], dims=["band", "y", "x"])
        none_rt = None

        with patch.dict("spectral_recovery.metrics.METRIC_FUNCS", {"yryr": y2r_mock}):

            compute_metrics(
                timeseries_data=valid_array,
                restoration_polygons=valid_frame,
                metrics=["YrYr"],
                recovery_target=none_rt
            )

            assert y2r_mock.call_args.kwargs["recovery_target"] is None
    
    def test_none_rt_with_metric_that_requires_rt_throws_value_err(
        self, valid_array, valid_frame
    ):
        y2r_mock = Mock()
        y2r_mock.return_value = xr.DataArray([[[0.0]]], dims=["band", "y", "x"])
        none_rt = None

        with patch.dict("spectral_recovery.metrics.METRIC_FUNCS", {"yryr": y2r_mock}):

            with pytest.raises(ValueError):
                compute_metrics(
                    timeseries_data=valid_array,
                    restoration_polygons=valid_frame,
                    metrics=["Y2R"],
                    recovery_target=none_rt
                )
            
            with pytest.raises(ValueError):
                compute_metrics(
                        timeseries_data=valid_array,
                        restoration_polygons=valid_frame,
                        metrics=["YrYr", "Y2R"],
                        recovery_target=none_rt
                    )

    def test_correct_metrics_called_from_metric_func_dict(
        self, valid_array, valid_frame, valid_rt
    ):
        patched_dict = {}
        multi_metrics = ["Y2R", "dNBR", "YrYr", "R80P"]
        for i, metric in enumerate(multi_metrics):
            metric_mock = Mock()
            metric_mock.return_value = xr.DataArray([[[i]]], dims=["band", "y", "x"])

            patched_dict[metric.lower()] = metric_mock

        with patch.dict("spectral_recovery.metrics.METRIC_FUNCS", patched_dict):

            compute_metrics(
                timeseries_data=valid_array,
                restoration_polygons=valid_frame,
                recovery_target=valid_rt,
                metrics=multi_metrics,
            )

        for m in multi_metrics:
            assert patched_dict[m.lower()].called_once()


    def test_output_data_array_stacked_along_metric_dim(
        self, valid_array, valid_frame, valid_rt
    ):

        patched_dict = {}
        multi_metrics = ["Y2R", "dNBR", "YrYr", "R80P"]
        for i, metric in enumerate(multi_metrics):
            metric_mock = Mock()
            metric_mock.return_value = xr.DataArray([[[i]]], dims=["band", "y", "x"])

            patched_dict[metric.lower()] = metric_mock

        with patch.dict("spectral_recovery.metrics.METRIC_FUNCS", patched_dict):

            result = compute_metrics(
                timeseries_data=valid_array,
                restoration_polygons=valid_frame,
                metrics=multi_metrics,
                recovery_target=valid_rt,
            )

        assert result[0].dims == ("metric", "band", "y", "x")
        assert sorted(result.metric.values) == sorted(multi_metrics)
        for i, metric in enumerate(multi_metrics):
            np.testing.assert_array_equal(
                result[0].sel(metric=metric).data, np.array([[[i]]])
            )

    def test_output_dataset_contains_all_polygons(
        self, valid_array, valid_rt
    ):
        multi_frame = gpd.GeoDataFrame(
            {
                "dist_start": [2012, 2011, 2012],
                "rest_start": [2013, 2012, 2013],
                "geometry": [self.valid_poly, self.valid_poly, self.valid_poly],
            },
            crs="EPSG:4326",
        )
        patched_dict = {}
        multi_metrics = ["Y2R", "dNBR"]
        for i, metric in enumerate(multi_metrics):
            metric_mock = Mock()
            metric_mock.return_value = xr.DataArray([[[i]]], dims=["band", "y", "x"])
            patched_dict[metric.lower()] = metric_mock
        
        expected_polyids = [0, 1, 2]

        with patch.dict("spectral_recovery.metrics.METRIC_FUNCS", patched_dict):

            result = compute_metrics(
                timeseries_data=valid_array,
                restoration_polygons=multi_frame,
                metrics=multi_metrics,
                recovery_target=valid_rt,
            )

        assert list(result.data_vars) == expected_polyids
        for polyid in expected_polyids:
            assert result[polyid].dims == ("metric", "band", "y", "x")
            for i, metric in enumerate(multi_metrics):
                np.testing.assert_array_equal(
                    result[polyid].sel(metric=metric).data, np.array([[[i]]])
                )

    def test_custom_params_passed_to_metric_funcs(
        self, valid_array, valid_frame, valid_rt
    ):
        metric = "Y2R"
        metric_mock = Mock()
        metric_mock.return_value = xr.DataArray([[[0.0]]], dims=["band", "y", "x"])

        with patch.dict(
            "spectral_recovery.metrics.METRIC_FUNCS", {metric.lower(): metric_mock}
        ):

            compute_metrics(
                timeseries_data=valid_array,
                restoration_polygons=valid_frame,
                recovery_target=valid_rt, 
                metrics=[metric],
                timestep=2,
                percent_of_target=60,
            )

            metric_mock.call_args.kwargs["params"]["timestep"] == 2
            metric_mock.call_args.kwargs["params"]["percent_of_target"] == 60
    
    def test_correct_params_passed_to_metric_func_if_rt_dict(self, valid_array):
        multi_frame = gpd.GeoDataFrame(
            {
                "dist_start": [2012, 2011, 2012],
                "rest_start": [2013, 2012, 2013],
                "geometry": [self.valid_poly, self.valid_poly, self.valid_poly],
            },
            crs="EPSG:4326",
        )
        patched_dict = {}
        multi_metrics = ["Y2R", "dNBR", "YrYr", "R80P"]
        for i, metric in enumerate(multi_metrics):
            metric_mock = Mock()
            metric_mock.return_value = xr.DataArray([[[i]]], dims=["band", "y", "x"])
            patched_dict[metric.lower()] = metric_mock
        clipped_array = valid_array.rio.clip(multi_frame["geometry"].values)

        rt_dict = {0: xr.DataArray([0]), 1: xr.DataArray([1]), 2: xr.DataArray([2])}
        with patch.dict("spectral_recovery.metrics.METRIC_FUNCS", patched_dict):

            compute_metrics(
                timeseries_data=valid_array,
                restoration_polygons=multi_frame,
                recovery_target=rt_dict,
                metrics=multi_metrics,
            )

        for m in multi_metrics:
            call0 = patched_dict[m.lower()].call_args_list[0].kwargs
            call1 = patched_dict[m.lower()].call_args_list[1].kwargs
            call2 = patched_dict[m.lower()].call_args_list[2].kwargs

            assert call0["restoration_start"] == 2013
            assert call1["restoration_start"] == 2012
            assert call2["restoration_start"] == 2013
            
            assert call0["disturbance_start"] == 2012
            assert call1["disturbance_start"] == 2011
            assert call2["disturbance_start"] == 2012

            xr.testing.assert_equal(call0["timeseries_data"], clipped_array)
            xr.testing.assert_equal(call1["timeseries_data"], clipped_array)
            xr.testing.assert_equal(call2["timeseries_data"], clipped_array)

            xr.testing.assert_equal(call0["recovery_target"], rt_dict[0])
            xr.testing.assert_equal(call1["recovery_target"], rt_dict[1])
            xr.testing.assert_equal(call2["recovery_target"], rt_dict[2])
    
    def test_correct_params_passed_to_metric_func_if_rt_not_dict(self, valid_array, valid_rt):
        multi_frame = gpd.GeoDataFrame(
            {
                "dist_start": [2012, 2011, 2012],
                "rest_start": [2013, 2012, 2013],
                "geometry": [self.valid_poly, self.valid_poly, self.valid_poly],
            },
            crs="EPSG:4326",
        )
        patched_dict = {}
        multi_metrics = ["Y2R", "dNBR", "YrYr", "R80P"]
        for i, metric in enumerate(multi_metrics):
            metric_mock = Mock()
            metric_mock.return_value = xr.DataArray([[[i]]], dims=["band", "y", "x"])
            patched_dict[metric.lower()] = metric_mock
        clipped_array = valid_array.rio.clip(multi_frame["geometry"].values)

        with patch.dict("spectral_recovery.metrics.METRIC_FUNCS", patched_dict):
            compute_metrics(
                timeseries_data=valid_array,
                restoration_polygons=multi_frame,
                recovery_target=valid_rt,
                metrics=multi_metrics,
            )

        for m in multi_metrics:
            call0 = patched_dict[m.lower()].call_args_list[0].kwargs
            call1 = patched_dict[m.lower()].call_args_list[1].kwargs
            call2 = patched_dict[m.lower()].call_args_list[2].kwargs

            assert call0["disturbance_start"] == 2012
            assert call1["disturbance_start"] == 2011
            assert call2["disturbance_start"] == 2012

            assert call0["restoration_start"] == 2013
            assert call1["restoration_start"] == 2012
            assert call2["restoration_start"] == 2013
            
            xr.testing.assert_equal(call0["timeseries_data"], clipped_array)
            xr.testing.assert_equal(call1["timeseries_data"], clipped_array)
            xr.testing.assert_equal(call2["timeseries_data"], clipped_array)

            xr.testing.assert_equal(call0["recovery_target"], valid_rt)
            xr.testing.assert_equal(call1["recovery_target"], valid_rt)
            xr.testing.assert_equal(call2["recovery_target"], valid_rt)



class TestY2R:
    valid_poly = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    
    @pytest.mark.parametrize(
        ("rt", "obs", "expected"),
        [
            (
                xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
                xr.DataArray(
                    [[[[70]], [[80]]]],  # meets recovery target in 1 year
                    coords={"time": [pd.to_datetime("2020"), pd.to_datetime("2021")]},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray([[[1.0]]], dims=["band", "y", "x"]),
            ),
            (
                xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
                xr.DataArray(
                    [[[[70]], [[90]]]],  # surpasses recovery target
                    coords={"time": [pd.to_datetime("2020"), pd.to_datetime("2021")]},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray([[[1.0]]], dims=["band", "y", "x"]),
            ),
            (
                xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
                xr.DataArray(
                    [[[[80]], [[90]]]],  # equals recovery target at start
                    coords={"time": [pd.to_datetime("2020"), pd.to_datetime("2021")]},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray([[[0.0]]], dims=["band", "y", "x"]),
            ),
            (
                xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
                xr.DataArray(
                    [[[[60]], [[70]]]],  # never meets recovery target
                    coords={"time": [pd.to_datetime("2020"), pd.to_datetime("2021")]},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray([[[-9999]]], dims=["band", "y", "x"]),
            ),
            (
                xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
                xr.DataArray(
                    [[
                        [[70, 60], [70, 60]],
                        [[80, 70], [70, 70]],
                        [[100, 70], [70, 80]],
                    ]],
                    coords={
                        "time": [
                            pd.to_datetime("2020"),
                            pd.to_datetime("2021"),
                            pd.to_datetime("2022"),
                        ]
                    },
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray(
                    [[[1.0, -9999], [-9999, 2.0]]], dims=["band", "y", "x"]
                ),
            ),
        ],
    )
    def test_single_target_y2r(self, rt, obs, expected):
        result = y2r(restoration_start=2020, timeseries_data=obs, recovery_target=rt).drop_vars('spatial_ref')
        assert result.equals(expected)

    def test_distinguishes_unrecovered_and_nan(self):
        rt = xr.DataArray([100], dims=["band"]).rio.write_crs("4326")
        obs = xr.DataArray(
            [[[[70, np.nan]], [[75, np.nan]], [[78, np.nan]]]],
            coords={
                "time": [
                    pd.to_datetime("2020"),
                    pd.to_datetime("2021"),
                    pd.to_datetime("2022"),
                ]
            },
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")

        expected = xr.DataArray([[[-9999, np.nan]]], dims=["band", "y", "x"]).rio.write_crs(
            "4326"
        )

        assert y2r(
            restoration_start=2020,
            timeseries_data=obs,
            recovery_target=rt
        ).equals(expected)
    
    def test_only_first_year_nan_returns_value(self):
        rt = xr.DataArray([100], dims=["band"]).rio.write_crs("4326")
        obs = xr.DataArray(
            [[[[np.nan]], [[75]], [[81]]]],
            coords={
                "time": [
                    pd.to_datetime("2020"),
                    pd.to_datetime("2021"),
                    pd.to_datetime("2022"),
                ]
            },
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")

        expected = xr.DataArray([[[2]]], dims=["band", "y", "x"]).rio.write_crs(
            "4326"
        )

        assert y2r(
            restoration_start=2020,
            timeseries_data=obs,
            recovery_target=rt
        ).equals(expected)

    def test_returns_first_recovered_year_when_successive_recovered_years_smaller(
        self
    ):
        rt = xr.DataArray([100], dims=["band"]).rio.write_crs("4326")
        obs = xr.DataArray(
            [[[[70]], [[90]], [[85]], [[80]], [[80]]]],
            coords={
                "time": [
                    pd.to_datetime("2020"),
                    pd.to_datetime("2021"),
                    pd.to_datetime("2022"),
                    pd.to_datetime("2023"),
                    pd.to_datetime("2024"),
                ]
            },
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")

        expected = xr.DataArray([[[1.0]]], dims=["band", "y", "x"]).rio.write_crs(
            "4326"
        )
        assert y2r(
            restoration_start=2020,
            timeseries_data=obs,
            recovery_target=rt
        ).equals(expected)

    def test_returns_first_recovered_year_when_successive_group_recovered_years_smaller(
        self,
    ):
        rt = xr.DataArray([100], dims=["band"]).rio.write_crs("4326")
        obs = xr.DataArray(
            [[[[70]], [[90]], [[95]], [[70]], [[70]], [[80]], [[80]]]],
            coords={
                "time": [
                    pd.to_datetime("2020"),
                    pd.to_datetime("2021"),
                    pd.to_datetime("2022"),
                    pd.to_datetime("2023"),
                    pd.to_datetime("2024"),
                    pd.to_datetime("2025"),
                    pd.to_datetime("2026"),
                ]
            },
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
      
        expected = xr.DataArray([[[1.0]]], dims=["band", "y", "x"]).rio.write_crs(
            "4326"
        )
        assert y2r(
            restoration_start=2020,
            timeseries_data=obs,
            recovery_target=rt
        ).equals(expected)

    @pytest.mark.parametrize(
        ("rt", "obs", "expected"),
        [
            (  # Meets one target, but not the other
                xr.DataArray([[[100, 80]]], dims=["band", "y", "x"]).rio.write_crs(
                    "4326"
                ),
                xr.DataArray(
                    [[[[70, 30]], [[80, 40]]]],
                    coords={"time": [pd.to_datetime("2020"), pd.to_datetime("2021")]},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray([[[1.0, -9999]]], dims=["band", "y", "x"]).rio.write_crs(
                    "4326"
                ),
            ),
            (  # Meets one target first year then meets next target second year
                xr.DataArray([[[100, 80]]], dims=["band", "y", "x"]).rio.write_crs(
                    "4326"
                ),
                xr.DataArray(
                    [[[[70, 30]], [[80, 40]], [[80, 65]]]],
                    coords={
                        "time": [
                            pd.to_datetime("2020"),
                            pd.to_datetime("2021"),
                            pd.to_datetime("2022"),
                        ]
                    },
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray([[[1.0, 2.0]]], dims=["band", "y", "x"]).rio.write_crs(
                    "4326"
                ),
            ),
        ],
    )
    def test_per_pixel_target(self, rt, obs, expected):

        assert y2r(
            restoration_start=2020,
            timeseries_data=obs,
            recovery_target=rt).equals(expected)

    @pytest.mark.parametrize(
        ("rt", "obs", "percent", "expected"),
        [
            (
                xr.DataArray([87], dims=["band"]).rio.write_crs("4326"),
                xr.DataArray(
                    [[[[80]], [[100]]]],
                    coords={"time": [pd.to_datetime("2020"), pd.to_datetime("2021")]},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                100,  # take full recovery target
                xr.DataArray([[[1.0]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
            ),
            (
                xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
                xr.DataArray(
                    [[[[10]], [[19]]]],
                    coords={"time": [pd.to_datetime("2020"), pd.to_datetime("2021")]},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                20,  # X percent of recovery target
                xr.DataArray([[[-9999]]], dims=["band", "y", "x"]).rio.write_crs(
                    "4326"
                ),
            ),
        ],
    )
    def test_percent_y2r(self, rt, obs, percent, expected):

        assert y2r(
            restoration_start=2020,
            timeseries_data=obs,
            recovery_target=rt,
            params={"percent_of_target": percent},
        ).equals(expected)
    
    def test_missing_years_in_recovery_window_throws_value_err(self):
        obs = xr.DataArray(
            [[[[10]], [[19]], [[20]]]],
            coords={"time": [pd.to_datetime("2020"), pd.to_datetime("2021"), pd.to_datetime("2023")]},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        rt = xr.DataArray([[[20]]], dims=["band", "y", "x"]).rio.write_crs("4326")
        percent = 100

        with pytest.raises(ValueError):
            y2r(
            restoration_start=2020,
            timeseries_data=obs,
            recovery_target=rt, 
            params={"percent_of_target": percent})
    
    def test_missing_years_outside_recovery_window_does_now_throw_value_err(self):
        obs = xr.DataArray(
            [[[[10]], [[19]], [[20]]]],
            coords={"time": [pd.to_datetime("2018"), pd.to_datetime("2020"), pd.to_datetime("2021")]},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        rt = xr.DataArray([[[20.0]]], dims=["band", "y", "x"]).rio.write_crs("4326")
        percent = 100
        expected = xr.DataArray([[[1.0]]], dims=["band", "y", "x"]).rio.write_crs("4326")

        assert y2r(
            restoration_start=2020,
            timeseries_data=obs,
            recovery_target=rt,
            params={"percent_of_target": percent},
        ).equals(expected)


class TestDNBR:
    year_period = [
        pd.to_datetime("2010"),
        pd.to_datetime("2011"),
        pd.to_datetime("2012"),
        pd.to_datetime("2013"),
        pd.to_datetime("2014"),
        pd.to_datetime("2015"),
    ]

    @pytest.mark.parametrize(
        ("obs","expected"),
        [
            (
                xr.DataArray(
                    [[[[50]], [[60]], [[70]], [[80]], [[90]], [[100]]]],
                    coords={"time": year_period},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),

                xr.DataArray(
                    [[[50]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
            (
                xr.DataArray(
                    [[
                        [[50, 10], [10, 20]],
                        [[60, 20], [10, 20]],
                        [[70, 30], [10, 20]],
                        [[80, 40], [10, 20]],
                        [[90, 50], [10, 20]],
                        [[100, 80], [10, 20]],
                    ]],
                    coords={"time": year_period},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray(
                    [[[50, 70], [0, 0]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
            (
                xr.DataArray(
                    [
                        [
                            [[50]],
                            [[60]],
                            [[70]],
                            [[80]],
                            [[90]],
                            [[100]],
                        ],
                        [
                            [[10]],
                            [[20]],
                            [[40]],
                            [[60]],
                            [[80]],
                            [[100]],
                        ],
                    ],
                    coords={"time": year_period},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray(
                    [[[50]], [[90]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
        ],
    )
    def test_default_dNBR(self, obs, expected):
        rest_start = 2010
        assert dnbr(
            restoration_start=rest_start,
            timeseries_data=obs
            ).equals(expected)

    def test_timestep_dNBR(self):
        rest_start = 2010
        obs = xr.DataArray(
            [[[[50]], [[60]], [[70]], [[80]], [[90]], [[100]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        timestep = 3

        expected = xr.DataArray(
            [[[30]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")

        assert dnbr(
            restoration_start=rest_start,
            timeseries_data=obs,
            params={"timestep": timestep},
        ).equals(expected)

    def test_invalid_timestep_throws_err(self):
        rest_start = 2010
        obs = xr.DataArray(
            [[[[50]], [[60]], [[70]], [[80]], [[90]], [[100]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        restoration_date = "2010"
        timestep = -2


        with pytest.raises(ValueError, match="timestep cannot be negative."):
            dnbr(
                restoration_start=rest_start,
                timeseries_data=obs,
                params={"timestep": timestep},
            )

    def test_timestep_too_large_throws_err(self):
        rest_start = 2010
        obs = xr.DataArray(
            [[[[50]], [[60]], [[70]], [[80]], [[90]], [[100]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        timestep = 6

        with pytest.raises(
            ValueError,
        ):
            dnbr(
                restoration_start=rest_start,
                timeseries_data=obs,
                params={"timestep": timestep},
            )


class TestRRI:
    year_period_RI = [
        pd.to_datetime("2000"),
        pd.to_datetime("2001"),
        pd.to_datetime("2002"),
        pd.to_datetime("2003"),
        pd.to_datetime("2004"),
        pd.to_datetime("2005"),
        pd.to_datetime("2006"),
    ]

    @pytest.mark.parametrize(
        ("obs", "expected"),
        [
            (  # max at t=4
                xr.DataArray(
                    [[[[70]], [[60]], [[70]], [[80]], [[90]], [[100]], [[80]]]],
                    coords={"time": year_period_RI},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray(
                    [[[4.0]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
            (  # max at t=5
                xr.DataArray(
                    [[[[70]], [[60]], [[70]], [[80]], [[100]], [[80]], [[100]]]],
                    coords={"time": year_period_RI},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray(
                    [[[4.0]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
        ],
    )
    def test_correct_default(
        self, obs, expected
    ):
        dist_start = 2000
        rest_start = 2001
        assert rri(
            disturbance_start=dist_start,
            restoration_start=rest_start,
            timeseries_data=obs
        ).equals(expected)

    def test_correct_multi_dimension_result(self):
        obs = xr.DataArray(
            [[
                [[50, 2], [30, 2]],  # dist_start
                [[20, 1], [25, 1]],  # dist_end
                [[20, 1.0], [20, 1.0]],
                [[30, 2], [15, 1]],
                [[40, 3], [20, 1]],
                [[50, 4], [25, 1]],
                [[50, 5], [30, 1]],
            ]],
            coords={"time": self.year_period_RI},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        dist_start = 2000
        rest_start = 2001


        # 4 pixels for dist_start - dist_end:
        # 1. 50 - 20 = 30
        # 2. 2 - 1 = 1
        # 3. 30 - 25 = 5
        # 4. 2 - 1 = 1
        # Max of t+5 and t+4:
        # 1. 50
        # 2. 5
        # 3. 30
        # 4. 1
        # t/f RRI will be:
        # 1. (50-20)/30 = 1
        # 2. (5-1)/1 = 4
        # 3. (30-25)/5 = 1
        # 4. 1-1/1 = 0
        expected = xr.DataArray(
            [[[1.0, 4.0], [1.0, 0.0]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")

        result = rri(
            disturbance_start=dist_start,
            restoration_start=rest_start,
            timeseries_data=obs
        )
        assert result.equals(expected)

    @pytest.mark.parametrize(
        ("obs", "timestep", "expected"),
        [
            (  # denom = 70 - 60 = 10, max of t+2 and t+1 = 120, 120-60 = 60, 60/10 = 6
                xr.DataArray(
                    [[[[70]], [[60]], [[120]], [[80]], [[90]], [[100]], [[80]]]],
                    coords={"time": year_period_RI},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                2,
                xr.DataArray(
                    [[[6.0]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
            (  # denom = 70 - 60 = 10, max of t+0 and t+1 = 70, 70-60 = 10, 10/10 = 1
                xr.DataArray(
                    [[[[70]], [[60]], [[70]], [[80]], [[90]], [[100]], [[80]]]],
                    coords={"time": year_period_RI},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                1,
                xr.DataArray(
                    [[[1.0]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
        ],
    )
    def test_timestep(
        self, obs, timestep, expected
    ):
        dist_start = 2000
        rest_start = 2001
        assert rri(
            disturbance_start=dist_start,
            restoration_start=rest_start,
            timeseries_data=obs,
            params={"timestep": timestep, "use_dist_avg": False},
        ).equals(expected)

    def test_neg_timestep_raises_err(self):
        dist_start = 2000
        rest_start = 2001
        obs = xr.DataArray([0])
        timestep = -1

        with pytest.raises(ValueError, match="timestep cannot be negative."):
            rri(
                disturbance_start=dist_start,
                restoration_start=rest_start,
                timeseries_data=obs,
                params={"timestep": timestep, "use_dist_avg": False},
            )


    def test_out_bound_timestep_raises_err(self):
        obs = xr.DataArray(
            [[[[70]], [[60]], [[70]], [[80]], [[90]], [[100]], [[80]]]],
            coords={"time": self.year_period_RI},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        rest_start = 2001
        dist_start = 2000
        timestep = 10

        with pytest.raises(
            ValueError,
        ):
            rri(
                disturbance_start=dist_start,
                restoration_start=rest_start,
                timeseries_data=obs,
                params={"timestep": timestep, "use_dist_avg": False},
            )

    def test_0_timestep_RRI_raises_err(self):
        obs = xr.DataArray(
            [[[[70]], [[60]], [[70]], [[80]], [[90]], [[100]], [[80]]]],
            coords={"time": self.year_period_RI},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        rest_start = 2001
        dist_start = 2000
        timestep = 0


        with pytest.raises(
            ValueError, match="timestep for RRI must be greater than 0."
        ):
            rri(
                disturbance_start=dist_start,
                restoration_start=rest_start,
                timeseries_data=obs,
                params={"timestep": timestep, "use_dist_avg": False},
            )

    def test_0_denom_sets_inf(self):
        obs = xr.DataArray(
            [[[[10]], [[10]], [[70]], [[80]], [[90]], [[100]], [[110]]]],
            coords={"time": self.year_period_RI},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        # Disturbance window of 10 - 10 = 0
        # Max of t+5 and t+4 is 110
        rest_start = 2001
        dist_start = 2000
        timestep = 5

        # 110 / 0 = nan --> check that we set it to inf
        expected = xr.DataArray(
            [[[np.inf]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")

        assert rri(
            disturbance_start=dist_start,
            restoration_start=rest_start,
            timeseries_data=obs,
            params={"timestep": timestep, "use_dist_avg": False},
        ).equals(expected)


class TestR80P:
    year_period = [
        pd.to_datetime("2010"),
        pd.to_datetime("2011"),
        pd.to_datetime("2012"),
        pd.to_datetime("2013"),
        pd.to_datetime("2014"),
        pd.to_datetime("2015"),
    ]

    @pytest.mark.parametrize(
        ("obs", "rt", "expected"),
        [
            (
                xr.DataArray(
                    [[[[40]], [[50]], [[60]], [[70]], [[80]], [[80]]]],
                    coords={"time": year_period},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray([100], dims=["band"]),
                xr.DataArray(
                    [[[1.0]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
            (
                xr.DataArray(
                    [[[[40]], [[50]], [[60]], [[70]], [[80]], [[120]]]],
                    coords={"time": year_period},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray([100], dims=["band"]),
                xr.DataArray(
                    [[[1.5]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
            (
                xr.DataArray(
                    [[[[40]], [[45]], [[50]], [[55]], [[60]], [[60]]]],
                    coords={"time": year_period},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray([100], dims=["band"]),
                xr.DataArray(
                    [[[0.75]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
        ],
    )
    def test_default_exactly_recovered(
        self, obs, rt, expected
    ):
        rest_start = 2010
        result = r80p(
            restoration_start=rest_start,
            timeseries_data=obs,
            recovery_target=rt
        )
        assert result.equals(expected)

    def test_timestep(self):
        obs = xr.DataArray(
            [[[[40]], [[50]], [[60]], [[70]], [[75]], [[80]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        rest_start = 2010
        timestep = 2
        rt = xr.DataArray([100], dims=["band"])

        expected = xr.DataArray(
            [[[0.75]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")

        result = r80p(
            restoration_start=rest_start,
            timeseries_data=obs,
            recovery_target=rt,
            params={"timestep": timestep, "percent_of_target": 80},
        )
        assert result.equals(expected)

    def test_percent(self):
        obs = xr.DataArray(
            [[[[40]], [[50]], [[60]], [[70]], [[75]], [[80]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        rest_start = 2010
        percent = 50
        rt = xr.DataArray([100], dims=["band"])

        expected = xr.DataArray(
            [[[1.6]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")

        result = r80p(
            restoration_start=rest_start,
            timeseries_data=obs,
            recovery_target=rt,
            params={"timestep": 5, "percent_of_target": percent},
        )
        assert result.equals(expected)

    def test_neg_timestep_value_err(self):
        obs = xr.DataArray(
            [[[[40]], [[50]], [[60]], [[70]], [[75]], [[80]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        rest_start = 2010
        rt = xr.DataArray([100], dims=["band"])
        neg_timestep = -1

        with pytest.raises(ValueError, match="timestep cannot be negative."):
            r80p(
                restoration_start=rest_start,
                timeseries_data=obs,
                recovery_target=rt,
                params={"timestep": neg_timestep, "percent_of_target": 80},
            )


class TestYrYr:
    year_period = [
        pd.to_datetime("2010"),
        pd.to_datetime("2011"),
        pd.to_datetime("2012"),
        pd.to_datetime("2013"),
        pd.to_datetime("2014"),
        pd.to_datetime("2015"),
    ]

    @pytest.mark.parametrize(
        ("obs", "expected"),
        [
            (  # Ri is greater than R0
                xr.DataArray(
                    [[[[40]], [[50]], [[60]], [[70]], [[80]], [[90]]]],
                    coords={"time": year_period},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray(
                    [[[10.0]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
            (  # Ri is less than R0
                xr.DataArray(
                    [[[[40]], [[50]], [[60]], [[50]], [[40]], [[30]]]],
                    coords={"time": year_period},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray(
                    [[[-2.0]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
            (  # Ri is equal than R0
                xr.DataArray(
                    [[[[40]], [[50]], [[60]], [[50]], [[40]], [[40]]]],
                    coords={"time": year_period},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),

                xr.DataArray(
                    [[[0.0]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
        ],
    )
    def test_default(self, obs, expected):
        rest_start = 2010
        result = yryr(
            restoration_start=rest_start,
            timeseries_data=obs,
        )
        assert result.equals(expected)

    def test_timestep(self):
        obs = xr.DataArray(
            [[[[40]], [[45]], [[50]], [[70]], [[80]], [[90]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        rest_start = 2010
        timestep = 2

        expected = xr.DataArray(
            [[[5.0]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")
        result = yryr(
            restoration_start=rest_start,
            timeseries_data=obs,
            params={"timestep": timestep},
        )
        assert result.equals(expected)

    def test_neg_timestep_throws_val_err(self):
        rest_start = 2010
        obs = xr.DataArray([0])
        timestep = -4

        with pytest.raises(ValueError, match="timestep cannot be negative."):
            yryr(
                restoration_start=rest_start,
            timeseries_data=obs,
                params={"timestep": timestep},
            )
