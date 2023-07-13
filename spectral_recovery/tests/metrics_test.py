import pytest
import xarray as xr
import numpy as np
import rioxarray

from spectral_recovery.metrics import (
    percent_recovered,
    years_to_recovery,
    dNBR,
    recovery_indicator,
)

# TODO: group tests into classes and add functions that check for CRS after computation


@pytest.mark.parametrize(
    ("baseline", "curr", "event", "expected"),
    [
        # TODO: make a func to construct xr dataarray to simplify parametrize call
        # see Xarray project testing modules for reference
        (
            xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
            xr.DataArray([[[80]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
            xr.DataArray([[[0]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
            xr.DataArray([[[0.8]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
        ),
        (
            xr.DataArray([100, 100], dims=["band"]).rio.write_crs("4326"),
            xr.DataArray([[[80]], [[50]]], dims=["band", "y", "x"]).rio.write_crs(
                "4326"
            ),
            xr.DataArray([[[0]], [[0]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
            xr.DataArray([[[0.8]], [[0.5]]], dims=["band", "y", "x"]).rio.write_crs(
                "4326"
            ),
        ),
        (
            xr.DataArray([100, 100], dims=["band"]).rio.write_crs("4326"),
            xr.DataArray(
                [[[80, 90], [100, 70]], [[50, 60], [70, 80]]], dims=["band", "y", "x"]
            ).rio.write_crs("4326"),
            xr.DataArray(
                [[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dims=["band", "y", "x"]
            ).rio.write_crs("4326"),
            xr.DataArray(
                [[[0.8, 0.9], [1.0, 0.7]], [[0.5, 0.6], [0.7, 0.8]]],
                dims=["band", "y", "x"],
            ).rio.write_crs("4326"),
        ),
        (
            xr.DataArray(
                [[[100, 90], [100, 70]], [[100, 60], [100, 80]]],
                dims=["band", "y", "x"],
            ).rio.write_crs("4326"),
            xr.DataArray(
                [[[80, 90], [100, 70]], [[50, 60], [70, 80]]], dims=["band", "y", "x"]
            ).rio.write_crs("4326"),
            xr.DataArray(
                [[[0, 0], [0, 0]], [[0, 0], [0, 0]]], dims=["band", "y", "x"]
            ).rio.write_crs("4326"),
            xr.DataArray(
                [[[0.8, 1.0], [1.0, 1.0]], [[0.5, 1.0], [0.7, 1.0]]],
                dims=["band", "y", "x"],
            ).rio.write_crs("4326"),
        ),
    ],
)
def test_correct_percent_recovered(baseline, curr, event, expected):
    assert percent_recovered(
        eval_stack=curr, baseline=baseline, event_obs=event
    ).equals(expected)


# TODO: revisit case #4
@pytest.mark.parametrize(
    ("baseline", "obs", "percent", "expected"),
    [
        (
            xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
            xr.DataArray(
                [[[[90]], [[100]]]],
                coords={"time": [np.datetime64("2020"), np.datetime64("2021")]},
                dims=["band", "time", "y", "x"],
            ).rio.write_crs("4326"),
            100,
            xr.DataArray([[[1.0]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
        ),
        (
            xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
            xr.DataArray(
                [[[[70]], [[80]]]],
                coords={"time": [np.datetime64("2020"), np.datetime64("2021")]},
                dims=["band", "time", "y", "x"],
            ).rio.write_crs("4326"),
            80,
            xr.DataArray([[[1.0]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
        ),
        (
            xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
            xr.DataArray(
                [[[[80]], [[90]]]],
                coords={"time": [np.datetime64("2020"), np.datetime64("2021")]},
                dims=["band", "time", "y", "x"],
            ).rio.write_crs("4326"),
            100,
            xr.DataArray([[[np.nan]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
        ),
        (
            xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
            xr.DataArray(
                [[[[80]], [[90]]]],
                coords={"time": [np.datetime64("2020"), np.datetime64("2021")]},
                dims=["band", "time", "y", "x"],
            ).rio.write_crs("4326"),
            100,
            xr.DataArray([[[np.nan]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
        ),
        (
            xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
            xr.DataArray(
                [[[[90]], [[100]], [[95]]]],
                coords={
                    "time": [
                        np.datetime64("2020"),
                        np.datetime64("2021"),
                        np.datetime64("2022"),
                    ]
                },
                dims=["band", "time", "y", "x"],
            ).rio.write_crs("4326"),
            100,
            xr.DataArray([[[np.nan]]], dims=["band", "y", "x"]).rio.write_crs("4326"),
        ),
        (
            xr.DataArray([100], dims=["band"]).rio.write_crs("4326"),
            xr.DataArray(
                [[[[90, 80], [70, 80]], [[100, 90], [80, 90]], [[100, 90], [90, 100]]]],
                coords={
                    "time": [
                        np.datetime64("2020"),
                        np.datetime64("2021"),
                        np.datetime64("2022"),
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
def test_correct_y2r(baseline, obs, percent, expected):
    assert years_to_recovery(
        image_stack=obs, baseline=baseline, percent=percent
    ).equals(expected)


# TODO: make this into func
year_period = [
    np.datetime64("2010"),
    np.datetime64("2011"),
    np.datetime64("2012"),
    np.datetime64("2013"),
    np.datetime64("2014"),
    np.datetime64("2015"),
]


@pytest.mark.parametrize(
    ("obs", "restoration_date", "trajectory_func", "expected"),
    [
        (
            xr.DataArray(
                [[[[50]], [[60]], [[70]], [[80]], [[90]], [[100]]]],
                coords={"time": year_period},
                dims=["band", "time", "y", "x"],
            ).rio.write_crs("4326"),
            "2010",
            None,
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
            None,
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
            None,
            xr.DataArray(
                [[[50]], [[90]]],
                dims=["band", "y", "x"],
            ).rio.write_crs("4326"),
        ),
    ],
)
def test_correct_dNBR(obs, restoration_date, trajectory_func, expected):
    assert dNBR(
        restoration_stack=obs,
        rest_start=restoration_date,
        trajectory_func=trajectory_func,
    ).equals(expected)


year_period_RI = [
    np.datetime64("2000"),
    np.datetime64("2001"),
    np.datetime64("2002"),
    np.datetime64("2003"),
    np.datetime64("2004"),
    np.datetime64("2005"),
    np.datetime64("2006"),
]

# TODO: follow-up on test cases with Melissa
@pytest.mark.parametrize(
    ("obs", "restoration_date", "trajectory_func", "expected"),
    [
        (
            xr.DataArray(
                [[[[70]], [[60]], [[70]], [[80]], [[90]], [[100]],[[110]]]],
                coords={"time": year_period_RI},
                dims=["band", "time", "y", "x"],
            ).rio.write_crs("4326"),
            "2001",
            None,
            xr.DataArray(
                [[[5.0]]],
                dims=["band", "y", "x"],
            ).rio.write_crs("4326"),
        ),
    ],
)
def test_correct_recovery_indicator(obs, restoration_date, trajectory_func, expected):
    print(recovery_indicator(
        image_stack=obs,
        rest_start=restoration_date,
        trajectory_func=trajectory_func,
    ).compute())
    assert recovery_indicator(
        image_stack=obs,
        rest_start=restoration_date,
        trajectory_func=trajectory_func,
    ).equals(expected)
