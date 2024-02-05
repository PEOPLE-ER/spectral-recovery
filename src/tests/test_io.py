import pytest
import xarray as xr
import pandas as pd
import numpy as np

from numpy.testing import assert_array_equal
from unittest.mock import patch
from spectral_recovery.enums import BandCommon, Index, Platform
from tests.utils import SAME_XR
from spectral_recovery.io.raster import (
    read_and_stack_tifs,
    _metrics_to_tifs,
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
        stacked_tifs = read_and_stack_tifs(
            path_to_tifs=tif_paths,
            platform=["Landsat-OLI"],
        )

        assert stacked_tifs.sizes["time"] == len(tif_paths)
        assert (
            stacked_tifs.sizes["band"]
            == mocked_rasterio_open.return_value.sizes["band"]
        )
        assert stacked_tifs.sizes["y"] == mocked_rasterio_open.return_value.sizes["y"]
        assert stacked_tifs.sizes["x"] == mocked_rasterio_open.return_value.sizes["x"]

    @pytest.mark.parametrize(
        ("filenames", "rasterio_return"),
        [
            (
                [f"202o"],
                xr.DataArray(
                    [[[[0]]]],
                    dims=["band", "time", "y", "x"],
                    coords={"band":[1]}
                ),
            ),
            (
                [f"tif2020"],
                xr.DataArray(
                    [[[[0]]]],
                    dims=["band", "time", "y", "x"],
                    coords={"band":[1]}
                ),
            ),
            (
                [f"20"],
                xr.DataArray(
                    [[[[0]]]],
                    dims=["band", "time", "y", "x"],
                    coords={"band":[1]}
                ),
            ),
            (
                [f"2020", f"not_a_year", f"2022"],
                xr.DataArray(
                    [[[[0]]]],
                    dims=["band", "time", "y", "x"],
                    coords={"band":[1]}
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
        rasterio_return,
    ):
        mocked_rasterio_open.return_value = rasterio_return
        with pytest.raises(
            ValueError,
        ):
            read_and_stack_tifs(path_to_tifs=filenames, platform=[])

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_correct_bands_from_tifs_with_long_name(self, mocked_rasterio_open):
        filenames = [f"path/to/2019.tif", f"path/to/2020.tif", f"path/to/2021.tif"]
        expected_bands = ["B", "R", "N"]
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band":[1, 2, 3]},
            attrs={"long_name": ["blue", "red", "nir"]},
        )
        mocked_rasterio_open.return_value = rasterio_return

        stacked_tifs = read_and_stack_tifs(
            path_to_tifs=filenames, platform=["Landsat-OLI"]
        )
        assert np.all(stacked_tifs["band"].data == expected_bands)

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_correct_bands_from_tifs_w_band_dict(self, mocked_rasterio_open):
        filenames = [f"path/to/2019.tif", f"path/to/2020.tif", f"path/to/2021.tif"]
        expected_bands = ["B", "R", "N"]
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band":[1, 2, 3]}
        )
        mocked_rasterio_open.return_value = rasterio_return

        stacked_tifs = read_and_stack_tifs(
            path_to_tifs=filenames,
            band_names={1: "blue", 2: "red", 3: "nir"},
            platform=["Landsat-OLI"],
        )

        assert np.all(stacked_tifs["band"].data == expected_bands)


    @patch(
        "rioxarray.open_rasterio",
    )
    def test_invalid_band_name_throws_error(self, mocked_rasterio_open):
        filenames = ["test_file"]
        rasterio_return = xr.DataArray(
            [[[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band":[1]}
        )
        mocked_rasterio_open.return_value = rasterio_return

        with pytest.raises(
            ValueError,
        ):
            stacked_tifs = read_and_stack_tifs(
                path_to_tifs=filenames,
                band_names={0: "not_a_band"},
                platform=["Landsat-OLI"],
            )


    @patch(
        "rioxarray.open_rasterio",
    )
    def test_band_dict_supersedes_band_desc(self, mocked_rasterio_open):
        filenames = [f"path/to/2019.tif", f"path/to/2020.tif", f"path/to/2021.tif"]
        expected_bands = ["B", "R", "N"]
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band":[1, 2, 3]},
            attrs={"long_name": ["swir", "green", "red"]},
        )
        mocked_rasterio_open.return_value = rasterio_return

        stacked_tifs = read_and_stack_tifs(
            path_to_tifs=filenames,
            band_names={1: "blue", 2: "red", 3: "nir"},
            platform=["Landsat-OLI"],
        )
        assert np.all(stacked_tifs["band"].data == expected_bands)

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_band_dict_assigns_name_by_key_not_order(self, mocked_rasterio_open):
        filenames = [f"path/to/2019.tif", f"path/to/2020.tif", f"path/to/2021.tif"]
        expected_bands = ["R", "B", "N"]
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band":[1, 2, 3]},
        )
        mocked_rasterio_open.return_value = rasterio_return

        stacked_tifs = read_and_stack_tifs(
            path_to_tifs=filenames,
            band_names={2: "blue", 1: "red", 3: "nir"},
            platform=["Landsat-OLI"],
        )
        # assert
        print(stacked_tifs["band"].data, expected_bands)
        assert_array_equal(stacked_tifs["band"].data, expected_bands)

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_band_dict_missing_mapping_throws_value_err(self, mocked_rasterio_open):
        filenames = [f"path/to/2019.tif", f"path/to/2020.tif", f"path/to/2021.tif"]
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band":[1, 2, 3]},
        )
        mocked_rasterio_open.return_value = rasterio_return

        with pytest.raises(
            ValueError,
        ):
            _ = read_and_stack_tifs(
                path_to_tifs=filenames,
                band_names={0: "red", 2: "nir"},
                platform=["Landsat-OLI"],
            )

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_band_dict_invalid_mapping_throws_value_err(self, mocked_rasterio_open):
        filenames = [f"path/to/2019.tif", f"path/to/2020.tif", f"path/to/2021.tif"]
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band":[1, 2, 3]},
        )
        mocked_rasterio_open.return_value = rasterio_return

        with pytest.raises(
            ValueError,
        ):
            _ = read_and_stack_tifs(
                path_to_tifs=filenames,
                band_names={0: "blue", 1: "red", 2: "nir", 3: "swir"},
                platform=["Landsat-OLI"],
            )

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_no_band_desc_or_band_names_throws_value_err(self, mocked_rasterio_open):
        filenames = [f"path/to/2019.tif", f"path/to/2020.tif", f"path/to/2021.tif"]
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band":[1, 2, 3]},
        )
        mocked_rasterio_open.return_value = rasterio_return

        with pytest.raises(
            ValueError,
        ):
            _ = read_and_stack_tifs(
                path_to_tifs=filenames,
                platform=["Landsat-OLI"],
            )

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
                    coords={"band":[1, 2, 3]},
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
        stacked_tifs = read_and_stack_tifs(
            path_to_tifs=filenames, platform=["Landsat-OLI"]
        )
        assert np.all(stacked_tifs["time"].data == sorted_years)

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_platform_is_added_to_attrs(self, mocked_rasterio_open):
        mocked_rasterio_open.return_value = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band":[1, 2, 3]},
            attrs={"long_name": ["blue", "red", "nir"]},
        )
        stacked_tifs = read_and_stack_tifs(
            path_to_tifs=[f"2017.tif", f"2018.tif", f"1992.tif", f"1990.tif"],
            platform=["Landsat-OLI", "Landsat-TM"],
        )
        assert stacked_tifs.attrs["platform"] == ["Landsat-OLI", "Landsat-TM"]