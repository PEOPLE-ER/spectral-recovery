import pytest
import xarray as xr
import numpy as np
import pandas as pd
import rioxarray

from spectral_recovery.restoration import RestorationArea
from spectral_recovery.metrics import (
    y2r,
    dnbr,
    rri,
    yryr,
    r80p,
    METRIC_FUNCS,
    compute_metrics
)


def test_metric_funcs_global_contains_all_funcs():
    expected_dict = {"y2r" : y2r, "dnbr": dnbr, "yryr": yryr, "r80p": r80p, "rri": rri}
    assert METRIC_FUNCS == expected_dict


class TestComputeMetrics:

    time_range = [str(x) for x in np.arange(2010, 2027)]
    baseline_array = xr.DataArray([[[1.0]], [[2.0]]])

    @pytest.fixture()
    def valid_resto_area(self):

        with rioxarray.open_rasterio(TIMESERIES_LEN_17, chunks="auto") as data:
            resto_poly = gpd.read_file(POLYGON_INBOUND)
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
        "metrics.y2r",
    )
    def test_Y2R_call_default(self, method_mock, valid_resto_area):
        method_mock.return_value = xr.DataArray([[1.0]], dims=["y", "x"])


#     @patch(
#         "spectral_recovery.metrics.yryr",
#     )
#     def test_YrYr_call_default(self, method_mock, valid_resto_area):
#         mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
#         method_mock.return_value = mocked_return

#         result = valid_resto_area.yryr()
#         expected_result = mocked_return.expand_dims(dim={"metric": [str(Metric.YRYR)]})

#         assert result.equals(expected_result)

#         timestep_default = 5

#         method_mock.assert_called_with(
#             image_stack=SAME_XR(valid_resto_area.restoration_image_stack),
#             rest_start=valid_resto_area.restoration_start,
#             timestep=timestep_default,
#         )

#     @patch(
#         "spectral_recovery.metrics.dnbr",
#     )
#     def test_dNBR_call_default(self, method_mock, valid_resto_area):
#         mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
#         method_mock.return_value = mocked_return

#         result = valid_resto_area.dnbr()
#         expected_result = mocked_return.expand_dims(dim={"metric": [str(Metric.DNBR)]})

#         assert result.equals(expected_result)

#         timestep_default = 5

#         method_mock.assert_called_with(
#             image_stack=SAME_XR(valid_resto_area.restoration_image_stack),
#             rest_start=valid_resto_area.restoration_start,
#             timestep=timestep_default,
#         )

#     @patch(
#         "spectral_recovery.metrics.rri",
#     )
#     def test_RRI_call_default(self, method_mock, valid_resto_area):
#         mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
#         method_mock.return_value = mocked_return

#         result = valid_resto_area._rri()
#         expected_result = mocked_return.expand_dims(dim={"metric": [str(Metric.RRI)]})

#         assert result.equals(expected_result)

#         timestep_default = 5

#         method_mock.assert_called_with(
#             image_stack=SAME_XR(valid_resto_area.restoration_image_stack),
#             rest_start=valid_resto_area.restoration_start,
#             dist_start=valid_resto_area.disturbance_start,
#             timestep=timestep_default,
#         )

#     @patch(
#         "spectral_recovery.metrics.r80p",
#     )
#     def test_R80P_call_default(self, method_mock, valid_resto_area):
#         mocked_return = xr.DataArray([[1.0]], dims=["y", "x"])
#         method_mock.return_value = mocked_return

#         result = valid_resto_area.r80p()
#         expected_result = mocked_return.expand_dims(dim={"metric": [str(Metric.R80P)]})

#         assert result.equals(expected_result)

#         percent_default = 80
#         timestep_default = 5

#         method_mock.assert_called_with(
#             image_stack=SAME_XR(valid_resto_area.restoration_image_stack),
#             rest_start=valid_resto_area.restoration_start,
#             recovery_target=SAME_XR(self.baseline_array),
#             timestep=timestep_default,
#             percent=percent_default,
#         )
    

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
                xr.DataArray([[[1.0]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
            ),
            (
                xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
                xr.DataArray(
                    [[[[70]], [[90]]]],  # surpasses recovery target
                    coords={"time": [pd.to_datetime("2020"), pd.to_datetime("2021")]},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray([[[1.0]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
            ),
            (
                xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
                xr.DataArray(
                    [[[[80]], [[90]]]],  # equals recovery target at start
                    coords={"time": [pd.to_datetime("2020"), pd.to_datetime("2021")]},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray([[[0.0]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
            ),
            (
                xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
                xr.DataArray(
                    [[[[60]], [[70]]]],  # never meets recovery target
                    coords={"time": [pd.to_datetime("2020"), pd.to_datetime("2021")]},
                    dims=["band", "time", "y", "x"],
                ).rio.write_crs("4326"),
                xr.DataArray([[[np.nan]]], dims=["band", "y", "x"]).rio.write_crs(
                    "4326"
                ),
            ),
            (
                xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
                xr.DataArray(
                    [
                        [
                            [[70, 60], [70, 60]],
                            [[80, 70], [70, 70]],
                            [[100, 70], [70, 80]],
                        ]
                    ],
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
                    [[[1.0, np.nan], [np.nan, 2.0]]], dims=["band", "y", "x"]
                ).rio.write_crs("4326"),
            ),
        ],
    )
    def test_single_target_y2r(self, recovery_target, obs, expected):
        assert y2r(
            image_stack=obs,
            recovery_target=recovery_target,
            rest_start="2020",
        ).equals(expected)

    def test_returns_first_recovered_year_when_successive_recovered_years_smaller(self):
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
        expected = xr.DataArray([[[1.0]]], dims=["band", "y", "x"]).rio.write_crs(
            "4326"
        )
        assert y2r(
            image_stack=obs,
            recovery_target=recovery_target,
            rest_start="2020",
        ).equals(expected)

    def test_returns_first_recovered_year_when_successive_group_recovered_years_smaller(
        self,
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
        expected = xr.DataArray([[[1.0]]], dims=["band", "y", "x"]).rio.write_crs(
            "4326"
        )
        assert y2r(
            image_stack=obs,
            recovery_target=recovery_target,
            rest_start="2020",
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
                xr.DataArray([[[1.0, np.nan]]], dims=["band", "y", "x"]).rio.write_crs(
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
    def test_per_pixel_target(self, recovery_target, obs, expected):
        assert y2r(
            image_stack=obs,
            recovery_target=recovery_target,
            rest_start="2020",
        ).equals(expected)

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
                xr.DataArray([[[np.nan]]], dims=["band", "y", "x"]).rio.write_crs(
                    "4326"
                ),
            ),
        ],
    )
    def test_percent_y2r(self, recovery_target, obs, percent, expected):
        assert y2r(
            image_stack=obs,
            recovery_target=recovery_target,
            percent=percent,
            rest_start="2020",
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
                    [
                        [
                            [[50, 10], [10, 20]],
                            [[60, 20], [10, 20]],
                            [[70, 30], [10, 20]],
                            [[80, 40], [10, 20]],
                            [[90, 50], [10, 20]],
                            [[100, 80], [10, 20]],
                        ]
                    ],
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
    def test_default_dNBR(self, obs, restoration_date, expected):
        assert dnbr(
            image_stack=obs,
            rest_start=restoration_date,
        ).equals(expected)

    def test_timestep_dNBR(self):
        obs = xr.DataArray(
            [[[[50]], [[60]], [[70]], [[80]], [[90]], [[100]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        restoration_date = "2010"
        timestep = 3

        expected = xr.DataArray(
            [[[30]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")

        assert dnbr(
            image_stack=obs,
            rest_start=restoration_date,
            timestep=timestep,
        ).equals(expected)

    def test_invalid_timestep_throws_err(self):
        obs = xr.DataArray(
            [[[[50]], [[60]], [[70]], [[80]], [[90]], [[100]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        restoration_date = "2010"
        timestep = -2

        with pytest.raises(ValueError, match="timestep cannot be negative."):
            dnbr(
                image_stack=obs,
                rest_start=restoration_date,
                timestep=timestep,
            )

    def test_timestep_too_large_throws_err(self):
        obs = xr.DataArray(
            [[[[50]], [[60]], [[70]], [[80]], [[90]], [[100]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        restoration_date = "2010"
        timestep = 6

        with pytest.raises(
            ValueError,
        ):
            dnbr(
                image_stack=obs,
                rest_start=restoration_date,
                timestep=timestep,
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
    def test_correct_default(self, obs, restoration_start, dist_start, expected):
        print(
            rri(
                image_stack=obs,
                rest_start=restoration_start,
                dist_start=dist_start,
            ),
            expected.data,
        )
        assert rri(
            image_stack=obs,
            rest_start=restoration_start,
            dist_start=dist_start,
        ).equals(expected)

    def test_correct_multi_dimension_result(self):
        obs = xr.DataArray(
            [
                [
                    [[50, 2], [30, 2]],
                    [[20, 1], [25, 1]],
                    [[20, 1.0], [20, 1.0]],
                    [[30, 2], [15, 1]],
                    [[40, 3], [20, 1]],
                    [[50, 4], [25, 1]],
                    [[50, 5], [30, 1]],
                ]
            ],
            coords={"time": self.year_period_RI},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        restoration_start = "2001"
        dist_start = "2000"
        timestep = 5
        # 4 pixels for dist_start - dist_end:
        # 1. 50 - 20 = 30
        # 2. 2 - 1 = 1
        # 3. 30 - 25 = 5
        # 4. 2 - 1 = 1
        # Max of t+5 and t+4:
        # 1. 60
        # 2. 5
        # 3. 30
        # 4. 1
        # t/f we want:
        # 1. (50-20)/30 = 1
        # 2. (5-1)/1 = 4
        # 3. (30-25)/5 = 1
        # 4. 1-1/1 = 0
        expected = xr.DataArray(
            [[[1.0, 4.0], [1.0, 0.0]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")
        result = rri(
            image_stack=obs,
            rest_start=restoration_start,
            dist_start=dist_start,
            timestep=timestep,
        )
        print(result, expected)
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
    def test_timestep(self, obs, restoration_start, dist_start, timestep, expected):
        assert rri(
            image_stack=obs,
            rest_start=restoration_start,
            dist_start=dist_start,
            timestep=timestep,
        ).equals(expected)

    def test_neg_timestep_raises_err(self):
        obs = xr.DataArray(
            [[[[70]], [[60]], [[70]], [[80]], [[90]], [[100]], [[80]]]],
            coords={"time": self.year_period_RI},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        restoration_start = "2001"
        dist_start = "2000"
        timestep = -1
        with pytest.raises(ValueError, match="timestep cannot be negative."):
            rri(
                image_stack=obs,
                rest_start=restoration_start,
                dist_start=dist_start,
                timestep=timestep,
            )

    def test_out_bound_timestep_raises_err(self):
        obs = xr.DataArray(
            [[[[70]], [[60]], [[70]], [[80]], [[90]], [[100]], [[80]]]],
            coords={"time": self.year_period_RI},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        restoration_start = "2001"
        dist_start = "2000"
        timestep = 10
        with pytest.raises(
            ValueError,
        ):
            rri(
                image_stack=obs,
                rest_start=restoration_start,
                dist_start=dist_start,
                timestep=timestep,
            )

    def test_0_timestep_RRI_raises_err(self):
        obs = xr.DataArray(
            [[[[70]], [[60]], [[70]], [[80]], [[90]], [[100]], [[80]]]],
            coords={"time": self.year_period_RI},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        restoration_start = "2001"
        dist_start = "2000"
        timestep = 0
        with pytest.raises(
            ValueError, match="timestep for RRI must be greater than 0."
        ):
            rri(
                image_stack=obs,
                rest_start=restoration_start,
                dist_start=dist_start,
                timestep=timestep,
            )

    def test_0_denom_sets_nan(self):
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
        # 110 / 0 = nan (not inf)
        expected = xr.DataArray(
            [[[np.nan]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")

        assert rri(
            image_stack=obs,
            rest_start=restoration_start,
            dist_start=dist_start,
            timestep=timestep,
        ).equals(expected)

    def test_use_dist_avg_uses_avg_of_dist(self):
        obs = xr.DataArray(
            [[[[80]], [[80]], [[60]], [[70]], [[50]], [[100]], [[120]]]],
            coords={"time": self.year_period_RI},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        restoration_start = "2004"
        dist_start = "2002"
        timestep = 2
        # Disturbance window of [60, 70, 50] has an average of 60
        # Previous disturbance window of [80, 80] has an average of 80
        # Maximum of t+2 and t+1 is 120
        # t/f (120 - 60) / (80 - 60) = 0.5
        expected = xr.DataArray(
            [[[3.0]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")

        result = rri(
            image_stack=obs,
            rest_start=restoration_start,
            dist_start=dist_start,
            timestep=timestep,
            use_dist_avg=True,
        )
        assert result.equals(expected)


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
        ("image_stack", "rest_start", "recovery_target", "expected"),
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
    def test_default_exactly_recovered(
        self, image_stack, rest_start, recovery_target, expected
    ):
        result = r80p(
            image_stack=image_stack,
            recovery_target=recovery_target,
            rest_start=rest_start,
        )
        assert result.equals(expected)

    def test_timestep(self):
        image_stack = xr.DataArray(
            [[[[40]], [[50]], [[60]], [[70]], [[75]], [[80]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        rest_start = "2010"
        timestep = 2
        recovery_target = xr.DataArray([100], dims=["band"])
        expected = xr.DataArray(
            [[[0.75]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")
        result = r80p(
            image_stack=image_stack,
            recovery_target=recovery_target,
            rest_start=rest_start,
            timestep=timestep,
        )
        assert result.equals(expected)

    def test_percent(self):
        image_stack = xr.DataArray(
            [[[[40]], [[50]], [[60]], [[70]], [[75]], [[80]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        rest_start = "2010"
        percent = 50
        recovery_target = xr.DataArray([100], dims=["band"])
        expected = xr.DataArray(
            [[[1.6]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")

        result = r80p(
            image_stack=image_stack,
            recovery_target=recovery_target,
            rest_start=rest_start,
            percent=percent,
        )
        assert result.equals(expected)

    def test_neg_timestep_value_err(self):
        image_stack = xr.DataArray(
            [[[[40]], [[50]], [[60]], [[70]], [[75]], [[80]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        restoration_date = "2010"
        recovery_target = xr.DataArray([100], dims=["band"])
        neg_timestep = -1

        with pytest.raises(ValueError, match="timestep cannot be negative."):
            r80p(
                image_stack=image_stack,
                recovery_target=recovery_target,
                rest_start=restoration_date,
                timestep=neg_timestep,
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
        ("image_stack", "rest_start", "expected"),
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
    def test_default(self, image_stack, rest_start, expected):
        result = yryr(
            image_stack=image_stack,
            rest_start=rest_start,
        )
        print(result, expected)
        assert result.equals(expected)

    def test_timestep(self):
        image_stack = xr.DataArray(
            [[[[40]], [[45]], [[50]], [[70]], [[80]], [[90]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        rest_start = "2010"
        timestep = 2
        expected = xr.DataArray(
            [[[5.0]]],
            dims=["band", "y", "x"],
        ).rio.write_crs("4326")
        result = yryr(
            image_stack=image_stack,
            rest_start=rest_start,
            timestep=timestep,
        )
        assert result.equals(expected)

    def test_neg_timestep_throws_val_err(self):
        image_stack = xr.DataArray(
            [[[[40]], [[45]], [[50]], [[70]], [[80]], [[90]]]],
            coords={"time": self.year_period},
            dims=["band", "time", "y", "x"],
        ).rio.write_crs("4326")
        rest_start = "2010"
        timestep = -4
        with pytest.raises(ValueError, match="timestep cannot be negative."):
            yryr(
                image_stack=image_stack,
                rest_start=rest_start,
                timestep=timestep,
            )
