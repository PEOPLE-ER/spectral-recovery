import pytest
import xarray as xr
import pandas as pd
import numpy as np


from unittest.mock import patch
from spectral_recovery.enums import BandCommon, Index
from spectral_recovery.io.raster import (
    read_and_stack_tifs,
    metrics_to_tifs,
)


class TestReadAndStackTifs:
    @pytest.mark.parametrize(
        ("tif_paths", "rasterio_return"),
        [
            (
                [f"a/fake/path/2020.tif"],
                xr.DataArray(
                    [[[[0]]]],
                    dims=["band", "time", "y", "x"],
                    attrs={"long_name": ["red"]},
                ),
            ),
            (
                [
                    f"a/fake/path/2020.tif",
                    f"a/fake/path/2021.tif",
                    f"a/fake/path/2022.tif",
                ],
                xr.DataArray(
                    [[[[0]]]],
                    dims=["band", "time", "y", "x"],
                    attrs={"long_name": ["red"]},
                ),
            ),
            (
                [
                    f"a/fake/path/2020.tif",
                    f"a/fake/path/2021.tif",
                    f"a/fake/path/2022.tif",
                ],
                xr.DataArray(
                    [[[[0]]], [[[0]]], [[[0]]]],
                    dims=["band", "time", "y", "x"],
                    attrs={"long_name": ["red", "blue", "nir"]},
                ),
            ),
            (
                [
                    f"a/fake/path/2020.tif",
                    f"a/fake/path/2021.tif",
                    f"a/fake/path/2022.tif",
                ],
                xr.DataArray(
                    [[[[0, 0], [0, 0]]]],
                    dims=["band", "time", "y", "x"],
                    attrs={"long_name": ["red"]},
                ),
            ),
            (
                [
                    f"a/fake/path/2020.tif",
                    f"a/fake/path/2021.tif",
                    f"a/fake/path/2022.tif",
                ],
                xr.DataArray(
                    [[[[0, 0], [0, 0]]], [[[0, 0], [0, 0]]], [[[0, 0], [0, 0]]]],
                    dims=["band", "time", "y", "x"],
                    attrs={"long_name": ["red", "blue", "nir"]},
                ),
            ),
        ],
    )
    @patch(
        "rioxarray.open_rasterio",
    )
    def test_correct_stacked_output_sizes(
        self,
        mocked_rasterio_open,
        tif_paths,
        rasterio_return,
    ):
        mocked_rasterio_open.return_value = rasterio_return
        stacked_tifs = read_and_stack_tifs(path_to_tifs=tif_paths)

        assert stacked_tifs.sizes["time"] == len(tif_paths)
        assert (
            stacked_tifs.sizes["band"]
            == mocked_rasterio_open.return_value.sizes["band"]
        )
        assert stacked_tifs.sizes["y"] == mocked_rasterio_open.return_value.sizes["y"]
        assert stacked_tifs.sizes["x"] == mocked_rasterio_open.return_value.sizes["x"]

    @pytest.mark.parametrize(
        ("filenames", "bad_name_index", "rasterio_return"),
        [
            (
                [f"202o"],
                0,
                xr.DataArray(
                    [[[[0]]]],
                    dims=["band", "time", "y", "x"],
                ),
            ),
            (
                [f"tif2020"],
                0,
                xr.DataArray(
                    [[[[0]]]],
                    dims=["band", "time", "y", "x"],
                ),
            ),
            (
                [f"20"],
                0,
                xr.DataArray(
                    [[[[0]]]],
                    dims=["band", "time", "y", "x"],
                ),
            ),
            (
                [f"2020", f"not_a_year", f"2022"],
                1,
                xr.DataArray(
                    [[[[0]]]],
                    dims=["band", "time", "y", "x"],
                ),
            ),
        ],
    )
    @patch(
        "rioxarray.open_rasterio",
    )
    def test_bad_filenames_throws_value_err(
        self,
        mocked_rasterio_open,
        filenames,
        bad_name_index,
        rasterio_return,
    ):
        mocked_rasterio_open.return_value = rasterio_return
        with pytest.raises(
            ValueError,
            match=(
                "TIF filenames must be in format 'YYYY' but recived:"
                f" '{filenames[bad_name_index]}'"
            ),
        ):
            read_and_stack_tifs(path_to_tifs=filenames)

    @pytest.mark.parametrize(
        ("expected_years", "filenames", "expected_bands", "rasterio_return"),
        [
            (
                [np.datetime64("2019"), np.datetime64("2020"), np.datetime64("2021")],
                [f"path/to/2019.tif", f"path/to/2020.tif", f"path/to/2021.tif"],
                [BandCommon.blue, BandCommon.red, BandCommon.nir],
                xr.DataArray(
                    [[[[0]]], [[[0]]], [[[0]]]],
                    dims=["band", "time", "y", "x"],
                    attrs={"long_name": ["blue", "red", "nir"]},
                ),
            ),
        ],
    )
    @patch(
        "rioxarray.open_rasterio",
    )
    def test_correct_coordinate_values(
        self,
        mocked_rasterio_open,
        expected_years,
        filenames,
        expected_bands,
        rasterio_return,
    ):
        mocked_rasterio_open.return_value = rasterio_return
        stacked_tifs = read_and_stack_tifs(path_to_tifs=filenames)
        assert np.all(stacked_tifs["band"].data == expected_bands)
        assert np.all(stacked_tifs["time"].data == expected_years)

    @pytest.mark.parametrize(
        ("sorted_years", "filenames", "rasterio_return"),
        [
            (
                [
                    np.datetime64("1990"),
                    np.datetime64("1992"),
                    np.datetime64("2017"),
                    np.datetime64("2018"),
                ],
                [f"2017.tif", f"2018.tif", f"1992.tif", f"1990.tif"],
                xr.DataArray(
                    [[[[0]]], [[[0]]], [[[0]]]],
                    dims=["band", "time", "y", "x"],
                    attrs={"long_name": ["blue", "red", "nir"]},
                ),
            ),
        ],
    )
    @patch(
        "rioxarray.open_rasterio",
    )
    def test_sorted_years(
        self,
        mocked_rasterio_open,
        sorted_years,
        filenames,
        rasterio_return,
    ):
        mocked_rasterio_open.return_value = rasterio_return
        stacked_tifs = read_and_stack_tifs(path_to_tifs=filenames)
        assert np.all(stacked_tifs["time"].data == sorted_years)

    # @patch(
    #     "rioxarray.open_rasterio",
    # )
    # def test_stacked_output_sizes(
    #     self, mocked_rasterio_open, tif_paths, rasterio_return,
    # ):
    #     mocked_rasterio_open.return_value = rasterio_return
    #     stacked_tifs = read_and_stack_tifs(path_to_tifs=tif_paths)

    #     assert stacked_tifs.sizes["time"] == len(tif_paths)
    #     assert (
    #         stacked_tifs.sizes["band"]
    #         == mocked_rasterio_open.return_value.sizes["band"]
    #     )
    #     assert stacked_tifs.sizes["y"] == mocked_rasterio_open.return_value.sizes["y"]
    #     assert stacked_tifs.sizes["x"] == mocked_rasterio_open.return_value.sizes["x"]
