import pytest
import xarray as xr
import numpy as np
import pandas as pd
import rioxarray

from spectral_recovery.metrics import (
    Y2R,
    dNBR,
    RRI,
    YrYr,
    R80P,
)

# TODO: group tests into classes and add functions that check for CRS after computation


# @pytest.mark.parametrize(
#     ("recovery_target", "curr", "event", "expected"),
#     [
#         # TODO: make a func to construct xr dataarray to simplify parametrize call
#         # see Xarray project testing modules for reference
#         (
#             xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
#             xr.DataArray([[[80]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
#             xr.DataArray([[[0]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
#             xr.DataArray([[[0.8]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
#         ),
#         (
#             xr.DataArray([100, 100], dims=["band"]).rio.write_crs("4326"),
#             xr.DataArray([[[80]], [[50]]], dims=["band", "y", "x"]).rio.write_crs(
#                 "4326"
#             ),
#             xr.DataArray([[[0]], [[0]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
#             xr.DataArray([[[0.8]], [[0.5]]], dims=["band", "y", "x"]).rio.write_crs(
#                 "4326"
#             ),
#         ),
#         (
#             xr.DataArray([100, 100], dims=["band"]).rio.write_crs("4326"),
#             xr.DataArray(
#                 [[[80, 90], [100, 70]], [[50, 60], [70, 80]]], dims=["band", "y", "x"]
#             ).rio.write_crs("4326"),
#             xr.DataArray(
#                 [[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dims=["band", "y", "x"]
#             ).rio.write_crs("4326"),
#             xr.DataArray(
#                 [[[0.8, 0.9], [1.0, 0.7]], [[0.5, 0.6], [0.7, 0.8]]],
#                 dims=["band", "y", "x"],
#             ).rio.write_crs("4326"),
#         ),
#         (
#             xr.DataArray(
#                 [[[100, 90], [100, 70]], [[100, 60], [100, 80]]],
#                 dims=["band", "y", "x"],
#             ).rio.write_crs("4326"),
#             xr.DataArray(
#                 [[[80, 90], [100, 70]], [[50, 60], [70, 80]]], dims=["band", "y", "x"]
#             ).rio.write_crs("4326"),
#             xr.DataArray(
#                 [[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dims=["band", "y", "x"]
#             ).rio.write_crs("4326"),
#             xr.DataArray(
#                 [[[0.8, 1.0], [1.0, 1.0]], [[0.5, 1.0], [0.7, 1.0]]],
#                 dims=["band", "y", "x"],
#             ).rio.write_crs("4326"),
#         ),
#     ],
# )
# def test_correct_percent_recovered(recovery_target, curr, event, expected):
#     assert percent_recovered(
#         eval_stack=curr, recovery_target=recovery_target, event_obs=event
#     ).equals(expected)


# TODO: revisit case #4
@pytest.mark.parametrize(
    ("recovery_target", "obs", "percent", "expected"),
    [
        (
            xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
            xr.DataArray(
                [[[[90]], [[100]]]],
                coords={"time": [pd.to_datetime("2020"), pd.to_datetime("2021")]},
                dims=["band", "time", "y", "x"],
            ).rio.write_crs("4326"),
            100,
            xr.DataArray([[[1.0]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
        ),
        (
            xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
            xr.DataArray(
                [[[[70]], [[80]]]],
                coords={"time": [pd.to_datetime("2020"), pd.to_datetime("2021")]},
                dims=["band", "time", "y", "x"],
            ).rio.write_crs("4326"),
            80,
            xr.DataArray([[[1.0]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
        ),
        (
            xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
            xr.DataArray(
                [[[[80]], [[90]]]],
                coords={"time": [pd.to_datetime("2020"), pd.to_datetime("2021")]},
                dims=["band", "time", "y", "x"],
            ).rio.write_crs("4326"),
            100,
            xr.DataArray([[[np.nan]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
        ),
        (
            xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
            xr.DataArray(
                [[[[80]], [[90]]]],
                coords={"time": [pd.to_datetime("2020"), pd.to_datetime("2021")]},
                dims=["band", "time", "y", "x"],
            ).rio.write_crs("4326"),
            100,
            xr.DataArray([[[np.nan]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
        ),
        (  # TODO: check this test with team
            xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
            xr.DataArray(
                [[[[90]], [[100]], [[95]]]],
                coords={
                    "time": [
                        pd.to_datetime("2020"),
                        pd.to_datetime("2021"),
                        pd.to_datetime("2022"),
                    ]
                },
                dims=["band", "time", "y", "x"],
            ).rio.write_crs("4326"),
            100,
            xr.DataArray([[[1.0]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
        ),
        (
            xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
            xr.DataArray(
                [[[[90, 80], [70, 80]], [[100, 90], [80, 90]], [[100, 90], [90, 100]]]],
                coords={
                    "time": [
                        pd.to_datetime("2020"),
                        pd.to_datetime("2021"),
                        pd.to_datetime("2022"),
                    ]
                },
                dims=["band", "time", "y", "x"],
            ).rio.write_crs("4326"),
            100,
            xr.DataArray(
                [[[1.0, np.nan], [np.nan, 2.0]]], dims=["band", "y", "x"]
            ).rio.write_crs("4326"),
        ),
    ],
)
def test_correct_y2r(recovery_target, obs, percent, expected):
    assert Y2R(
        image_stack=obs,
        recovery_target=recovery_target,
        percent=percent,
        rest_start="2020",
    ).equals(expected)


# TODO: make this into func
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
def test_correct_dNBR(obs, restoration_date, expected):
    assert dNBR(
        image_stack=obs,
        rest_start=restoration_date,
    ).equals(expected)


year_period_RI = [
    pd.to_datetime("2000"),
    pd.to_datetime("2001"),
    pd.to_datetime("2002"),
    pd.to_datetime("2003"),
    pd.to_datetime("2004"),
    pd.to_datetime("2005"),
    pd.to_datetime("2006"),
]


# TODO: follow-up on test cases with Melissa
@pytest.mark.parametrize(
    ("obs", "restoration_date", "expected"),
    [
        (
            xr.DataArray(
                [[[[70]], [[60]], [[70]], [[80]], [[90]], [[100]], [[110]]]],
                coords={"time": year_period_RI},
                dims=["band", "time", "y", "x"],
            ).rio.write_crs("4326"),
            "2001",
            xr.DataArray(
                [[[5.0]]],
                dims=["band", "y", "x"],
            ).rio.write_crs("4326"),
        ),
    ],
)
def test_correct_RRI(obs, restoration_date, expected):
    assert RRI(
        image_stack=obs,
        rest_start=restoration_date,
    ).equals(expected)
