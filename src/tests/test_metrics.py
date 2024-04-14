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
                "reference_start": [2010],
                "reference_end": [2010],
                "geometry": [self.valid_poly],
            },
            crs="EPSG:4326",
        )
        return valid_frame

    @pytest.fixture()
    def valid_rt(self, valid_array):

        valid_rt = valid_array[:,0, :, :].drop_vars("time")
        return valid_rt

    @patch("spectral_recovery.metrics.RestorationArea")
    def test_ra_built_with_given_polys(
        self, ra_mock, valid_array, valid_frame, valid_rt
    ):
        ra_mock.return_value = "ra_return"
        y2r_mock = Mock()
        y2r_mock.return_value = xr.DataArray([[[0.0]]], dims=["band", "y", "x"])

        with patch.dict("spectral_recovery.metrics.METRIC_FUNCS", {"y2r": y2r_mock}):

            compute_metrics(
                timeseries_data=valid_array,
                restoration_polygons=valid_frame,
                metrics=["Y2R"],
                recovery_target=valid_rt
            )

            pd.testing.assert_frame_equal(
                ra_mock.call_args.kwargs["restoration_polygon"], valid_frame
            )
            xr.testing.assert_equal(
                ra_mock.call_args.kwargs["composite_stack"], valid_array
            )
            xr.testing.assert_equal(
                ra_mock.call_args.kwargs["recovery_target"], valid_rt
            )

    # @patch("spectral_recovery.metrics.compute_indices")
    # @patch("spectral_recovery.metrics.RestorationArea")
    # def test_ra_built_with_reference_polys(self, ra_mock, indices_mock, valid_array, valid_frame):
    #     indices_mock.return_value = "indices_return"
    #     ra_mock.return_value = "ra_return"
    #     y2r_mock = Mock()
    #     y2r_mock.return_value = xr.DataArray([[[0.0]]], dims=["band", "y", "x"])

    #     with patch.dict("spectral_recovery.metrics.METRIC_FUNCS", {"y2r": y2r_mock}):

    #         compute_metrics(
    #             timeseries_data=valid_array,
    #             restoration_polygons=valid_frame,
    #             reference_polygons=___
    #             metrics=["Y2R"],
    #         )

    #         pd.testing.assert_frame_equal(ra_mock.call_args.kwargs["reference_polygons"], valid_frame)
    #         assert ra_mock.call_args.kwargs["composite_stack"] == "indices_return"
    #         assert isinstance(ra_mock.call_args.kwargs["recovery_target_method"], MedianTarget)
    #         assert ra_mock.call_args.kwargs["recovery_target_method"].scale == "polygon"

    @patch("spectral_recovery.metrics.RestorationArea")
    def test_correct_metrics_called_from_metric_func_dict(
        self, ra_mock, valid_array, valid_frame, valid_rt
    ):
        ra_mock.return_value = "ra_return"

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

        for metric_mock_func in patched_dict.values():
            metric_mock_func.assert_called_with(
                ra="ra_return", params={"timestep": 5, "percent_of_target": 80}
            )

    @patch("spectral_recovery.metrics.RestorationArea")
    def test_output_data_stacked_along_metric_dim(
        self, ra_mock, valid_array, valid_frame, valid_rt
    ):
        ra_mock.return_value = "ra_return"

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

        assert result.dims == ("metric", "band", "y", "x")
        assert sorted(result.metric.values) == sorted(multi_metrics)
        for i, metric in enumerate(multi_metrics):
            np.testing.assert_array_equal(
                result.sel(metric=metric).data, np.array([[[i]]])
            )

    @patch("spectral_recovery.metrics.RestorationArea")
    def test_custom_params_passed_to_metric_funcs(
        self, ra_mock, valid_array, valid_frame, valid_rt
    ):
        ra_mock.return_value = "ra_return"
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


class TestY2R:
    @pytest.mark.parametrize(
        ("recovery_target", "obs", "expected"),
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
    @patch("spectral_recovery.restoration.RestorationArea")
    def test_single_target_y2r(self, ra_mock, recovery_target, obs, expected):
        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = "2020"
        ra_mock.recovery_target = recovery_target

        result = y2r(ra=ra_mock).drop_vars('spatial_ref')
        print(result, expected)
        assert result.equals(expected)

    @patch("spectral_recovery.restoration.RestorationArea")
    def test_distinguishes_unrecovered_and_nan(self, ra_mock):
        recovery_target = xr.DataArray([100], dims=["band"]).rio.write_crs("4326")
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
        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = "2020"
        ra_mock.recovery_target = recovery_target

        expected = xr.DataArray([[[-9999, np.nan]]], dims=["band", "y", "x"]).rio.write_crs(
            "4326"
        )

        assert y2r(
            ra=ra_mock
        ).equals(expected)
    
    @patch("spectral_recovery.restoration.RestorationArea")
    def test_only_first_year_nan_returns_value(self, ra_mock):
        recovery_target = xr.DataArray([100], dims=["band"]).rio.write_crs("4326")
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
        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = "2020"
        ra_mock.recovery_target = recovery_target

        expected = xr.DataArray([[[2]]], dims=["band", "y", "x"]).rio.write_crs(
            "4326"
        )

        print(y2r(
            ra=ra_mock
        ), expected)

        assert y2r(
            ra=ra_mock
        ).equals(expected)

    @patch("spectral_recovery.restoration.RestorationArea")
    def test_returns_first_recovered_year_when_successive_recovered_years_smaller(
        self, ra_mock
    ):
        recovery_target = xr.DataArray([100], dims=["band"]).rio.write_crs("4326")
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
        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = "2020"
        ra_mock.recovery_target = recovery_target

        expected = xr.DataArray([[[1.0]]], dims=["band", "y", "x"]).rio.write_crs(
            "4326"
        )
        assert y2r(
            ra=ra_mock,
        ).equals(expected)

    @patch("spectral_recovery.restoration.RestorationArea")
    def test_returns_first_recovered_year_when_successive_group_recovered_years_smaller(
        self,
        ra_mock,
    ):
        recovery_target = xr.DataArray([100], dims=["band"]).rio.write_crs("4326")
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
        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = "2020"
        ra_mock.recovery_target = recovery_target
        expected = xr.DataArray([[[1.0]]], dims=["band", "y", "x"]).rio.write_crs(
            "4326"
        )
        assert y2r(
            ra=ra_mock,
        ).equals(expected)

    @pytest.mark.parametrize(
        ("recovery_target", "obs", "expected"),
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
    @patch("spectral_recovery.restoration.RestorationArea")
    def test_per_pixel_target(self, ra_mock, recovery_target, obs, expected):
        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = "2020"
        ra_mock.recovery_target = recovery_target

        assert y2r(ra=ra_mock).equals(expected)

    @pytest.mark.parametrize(
        ("recovery_target", "obs", "percent", "expected"),
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
    @patch("spectral_recovery.restoration.RestorationArea")
    def test_percent_y2r(self, ra_mock, recovery_target, obs, percent, expected):
        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = "2020"
        ra_mock.recovery_target = recovery_target

        assert y2r(
            ra=ra_mock,
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
        ("obs", "restoration_date", "expected"),
        [
            (
                xr.DataArray(
                    [[[[50]], [[60]], [[70]], [[80]], [[90]], [[100]]]],
                    coords={"time": year_period},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                "2010",
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
                "2010",
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
                "2010",
                xr.DataArray(
                    [[[50]], [[90]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
        ],
    )
    @patch("spectral_recovery.restoration.RestorationArea")
    def test_default_dNBR(self, ra_mock, obs, restoration_date, expected):
        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = restoration_date
        ra_mock.timeseries_end = "2015"

        assert dnbr(ra=ra_mock).equals(expected)

    @patch("spectral_recovery.restoration.RestorationArea")
    def test_timestep_dNBR(self, ra_mock):
        obs = xr.DataArray(
            [[[[50]], [[60]], [[70]], [[80]], [[90]], [[100]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        restoration_date = "2010"
        timestep = 3

        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = restoration_date
        ra_mock.timeseries_end = "2015"

        expected = xr.DataArray(
            [[[30]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")

        assert dnbr(
            ra=ra_mock,
            params={"timestep": timestep},
        ).equals(expected)

    @patch("spectral_recovery.restoration.RestorationArea")
    def test_invalid_timestep_throws_err(self, ra_mock):
        obs = xr.DataArray(
            [[[[50]], [[60]], [[70]], [[80]], [[90]], [[100]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        restoration_date = "2010"
        timestep = -2

        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = restoration_date
        ra_mock.timeseries_end = "2015"

        with pytest.raises(ValueError, match="timestep cannot be negative."):
            dnbr(
                ra=ra_mock,
                params={"timestep": timestep},
            )

    @patch("spectral_recovery.restoration.RestorationArea")
    def test_timestep_too_large_throws_err(self, ra_mock):
        obs = xr.DataArray(
            [[[[50]], [[60]], [[70]], [[80]], [[90]], [[100]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        restoration_date = "2010"
        timestep = 6

        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = restoration_date
        ra_mock.timeseries_end = "2015"

        with pytest.raises(
            ValueError,
        ):
            dnbr(
                ra=ra_mock,
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
        ("obs", "restoration_start", "dist_start", "expected"),
        [
            (  # max at t=4
                xr.DataArray(
                    [[[[70]], [[60]], [[70]], [[80]], [[90]], [[100]], [[80]]]],
                    coords={"time": year_period_RI},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                "2001",
                "2000",
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
                "2001",
                "2000",
                xr.DataArray(
                    [[[4.0]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
        ],
    )
    @patch("spectral_recovery.restoration.RestorationArea")
    def test_correct_default(
        self, ra_mock, obs, restoration_start, dist_start, expected
    ):
        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = restoration_start
        ra_mock.disturbance_start = dist_start

        assert rri(ra=ra_mock).equals(expected)

    @patch("spectral_recovery.restoration.RestorationArea")
    def test_correct_multi_dimension_result(self, ra_mock):
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
        restoration_start = "2001"
        dist_start = "2000"

        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = restoration_start
        ra_mock.disturbance_start = dist_start

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
            ra=ra_mock,
        )

        assert result.equals(expected)

    @pytest.mark.parametrize(
        ("obs", "restoration_start", "dist_start", "timestep", "expected"),
        [
            (  # denom = 70 - 60 = 10, max of t+2 and t+1 = 120, 120-60 = 60, 60/10 = 6
                xr.DataArray(
                    [[[[70]], [[60]], [[120]], [[80]], [[90]], [[100]], [[80]]]],
                    coords={"time": year_period_RI},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                "2001",
                "2000",
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
                "2001",
                "2000",
                1,
                xr.DataArray(
                    [[[1.0]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
        ],
    )
    @patch("spectral_recovery.restoration.RestorationArea")
    def test_timestep(
        self, ra_mock, obs, restoration_start, dist_start, timestep, expected
    ):
        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = restoration_start
        ra_mock.disturbance_start = dist_start

        assert rri(
            ra=ra_mock,
            params={"timestep": timestep, "use_dist_avg": False},
        ).equals(expected)

    @patch("spectral_recovery.restoration.RestorationArea")
    def test_neg_timestep_raises_err(self, ra_mock):
        # give an empty RA mock because this error should be thrown first
        # before any processing/prep is done in the function
        timestep = -1

        with pytest.raises(ValueError, match="timestep cannot be negative."):
            rri(
                ra=ra_mock,
                params={"timestep": timestep, "use_dist_avg": False},
            )

    @patch("spectral_recovery.restoration.RestorationArea")
    def test_out_bound_timestep_raises_err(self, ra_mock):
        obs = xr.DataArray(
            [[[[70]], [[60]], [[70]], [[80]], [[90]], [[100]], [[80]]]],
            coords={"time": self.year_period_RI},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        restoration_start = "2001"
        dist_start = "2000"
        timestep = 10

        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = restoration_start
        ra_mock.disturbance_start = dist_start

        with pytest.raises(
            ValueError,
        ):
            rri(
                ra=ra_mock,
                params={"timestep": timestep, "use_dist_avg": False},
            )

    @patch("spectral_recovery.restoration.RestorationArea")
    def test_0_timestep_RRI_raises_err(self, ra_mock):
        obs = xr.DataArray(
            [[[[70]], [[60]], [[70]], [[80]], [[90]], [[100]], [[80]]]],
            coords={"time": self.year_period_RI},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        restoration_start = "2001"
        dist_start = "2000"
        timestep = 0

        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = restoration_start
        ra_mock.disturbance_start = dist_start

        with pytest.raises(
            ValueError, match="timestep for RRI must be greater than 0."
        ):
            rri(
                ra=ra_mock,
                params={"timestep": timestep, "use_dist_avg": False},
            )

    @patch("spectral_recovery.restoration.RestorationArea")
    def test_0_denom_sets_inf(self, ra_mock):
        obs = xr.DataArray(
            [[[[10]], [[10]], [[70]], [[80]], [[90]], [[100]], [[110]]]],
            coords={"time": self.year_period_RI},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        # Disturbance window of 10 - 10 = 0
        # Max of t+5 and t+4 is 110
        restoration_start = "2001"
        dist_start = "2000"
        timestep = 5

        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = restoration_start
        ra_mock.disturbance_start = dist_start

        # 110 / 0 = nan (not inf)
        expected = xr.DataArray(
            [[[np.inf]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")

        assert rri(
            ra=ra_mock,
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
        ("obs", "rest_start", "recovery_target", "expected"),
        [
            (
                xr.DataArray(
                    [[[[40]], [[50]], [[60]], [[70]], [[80]], [[80]]]],
                    coords={"time": year_period},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                "2010",
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
                "2010",
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
                "2010",
                xr.DataArray([100], dims=["band"]),
                xr.DataArray(
                    [[[0.75]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
        ],
    )
    @patch("spectral_recovery.restoration.RestorationArea")
    def test_default_exactly_recovered(
        self, ra_mock, obs, rest_start, recovery_target, expected
    ):
        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = rest_start
        ra_mock.recovery_target = recovery_target

        result = r80p(ra=ra_mock)
        assert result.equals(expected)

    @patch("spectral_recovery.restoration.RestorationArea")
    def test_timestep(self, ra_mock):
        obs = xr.DataArray(
            [[[[40]], [[50]], [[60]], [[70]], [[75]], [[80]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        rest_start = "2010"
        timestep = 2
        recovery_target = xr.DataArray([100], dims=["band"])

        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = rest_start
        ra_mock.recovery_target = recovery_target

        expected = xr.DataArray(
            [[[0.75]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")

        result = r80p(
            ra=ra_mock,
            params={"timestep": timestep, "percent_of_target": 80},
        )
        assert result.equals(expected)

    @patch("spectral_recovery.restoration.RestorationArea")
    def test_percent(self, ra_mock):
        obs = xr.DataArray(
            [[[[40]], [[50]], [[60]], [[70]], [[75]], [[80]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        rest_start = "2010"
        percent = 50
        recovery_target = xr.DataArray([100], dims=["band"])
        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = rest_start
        ra_mock.recovery_target = recovery_target

        expected = xr.DataArray(
            [[[1.6]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")

        result = r80p(
            ra=ra_mock,
            params={"timestep": 5, "percent_of_target": percent},
        )
        assert result.equals(expected)

    @patch("spectral_recovery.restoration.RestorationArea")
    def test_neg_timestep_value_err(self, ra_mock):
        obs = xr.DataArray(
            [[[[40]], [[50]], [[60]], [[70]], [[75]], [[80]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        restoration_date = "2010"
        recovery_target = xr.DataArray([100], dims=["band"])
        neg_timestep = -1
        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = restoration_date
        ra_mock.recovery_target = recovery_target

        with pytest.raises(ValueError, match="timestep cannot be negative."):
            r80p(
                ra=ra_mock,
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
        ("obs", "rest_start", "expected"),
        [
            (  # Ri is greater than R0
                xr.DataArray(
                    [[[[40]], [[50]], [[60]], [[70]], [[80]], [[90]]]],
                    coords={"time": year_period},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                "2010",
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
                "2010",
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
                "2010",
                xr.DataArray(
                    [[[0.0]]],
                    dims=["band", "y", "x"],
                ).rio.write_crs("4326"),
            ),
        ],
    )
    @patch("spectral_recovery.restoration.RestorationArea")
    def test_default(self, ra_mock, obs, rest_start, expected):
        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = rest_start
        result = yryr(
            ra=ra_mock,
        )
        assert result.equals(expected)

    @patch("spectral_recovery.restoration.RestorationArea")
    def test_timestep(self, ra_mock):
        obs = xr.DataArray(
            [[[[40]], [[45]], [[50]], [[70]], [[80]], [[90]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        rest_start = "2010"
        timestep = 2
        ra_mock.restoration_image_stack = obs
        ra_mock.restoration_start = rest_start

        expected = xr.DataArray(
            [[[5.0]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")
        result = yryr(
            ra=ra_mock,
            params={"timestep": timestep},
        )
        assert result.equals(expected)

    @patch("spectral_recovery.restoration.RestorationArea")
    def test_neg_timestep_throws_val_err(self, ra_mock):
        timestep = -4

        with pytest.raises(ValueError, match="timestep cannot be negative."):
            yryr(
                ra=ra_mock,
                params={"timestep": timestep},
            )
