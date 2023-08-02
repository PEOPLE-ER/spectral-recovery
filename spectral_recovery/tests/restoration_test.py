import pytest
import xarray as xr
import numpy as np
import rioxarray
import geopandas as gpd
import pandas as pd

from mock import patch
from spectral_recovery.restoration import ReferenceSystem, RestorationArea
from spectral_recovery.enums import Metric


DATETIME_FREQ = "YS"  # TODO: should this be kept somewhere else in the project? Seem wrong that it's defined again here and in timeseries


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
def test_RestorationArea_initialization(
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
def test_RestorationArea_not_contained_error(
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
        print(time_range)
        stack = stack.assign_coords(
            time=(pd.date_range(time_range[0], time_range[-1], freq=DATETIME_FREQ))
        )
        resto_poly = gpd.read_file(resto_poly)
        with pytest.raises(
            ValueError,
            match="RestorationArea not contained by stack. Better message soon!",
        ):
            resto_a = RestorationArea(
                restoration_polygon=resto_poly,
                restoration_year=resto_year,
                reference_system=ref_sys,
                composite_stack=stack,
                end_year=end_year,
            )


@pytest.fixture()
def valid_resto_area():
    polygon = "1p_test.gpkg"
    restoration_year = (pd.to_datetime("2015"),)
    reference_year = (pd.to_datetime("2012"),)
    raster = "test_recovered.tif"
    time_range = [str(x) for x in np.arange(2010, 2022)]

    with rioxarray.open_rasterio(raster, chunks="auto") as data:
        resto_poly = gpd.read_file(polygon)
        stack = data
        stack = stack.rename({"band": "time"})
        stack = stack.assign_coords(
            time=(pd.date_range(time_range[0], time_range[-1], freq=DATETIME_FREQ))
        )
        # print(stack)
        resto_area = RestorationArea(
            restoration_polygon=resto_poly,
            restoration_year=restoration_year[0],
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
def test_RestorationArea_metric_switch_case(
    mock_stack, valid_resto_area, metric_list, metric_method, mocker
):
    metric_patch = mocker.patch(f"spectral_recovery.restoration.{metric_method}")

    valid_resto_area.metrics(metric_list)
    
    assert metric_patch.call_count == 1
    assert mock_stack.call_count == 1


@patch("spectral_recovery.restoration._stack_bands")
def test_RestorationArea_multiple_metrics(
    mock_stack, valid_resto_area, mocker
):
    metrics_list = [Metric.percent_recovered, Metric.years_to_recovery, Metric.recovery_indicator, Metric.dNBR]
    methods_list = ["percent_recovered", "years_to_recovery", "recovery_indicator", "dNBR"]

    patches = {}
    for i, metric in enumerate(metrics_list):
        patches[metric] = mocker.patch(f"spectral_recovery.restoration.{methods_list[i]}")

    valid_resto_area.metrics(metrics_list)
    
    for i, metric in enumerate(metrics_list):
        assert patches[metric].call_count == 1
    assert mock_stack.call_count == 1


def test_RestorationArea_stacked_metrics():
    pass