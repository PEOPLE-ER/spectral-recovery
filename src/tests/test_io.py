import pytest
import dask.array as da

import xarray as xr
import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal
from unittest.mock import patch
from spectral_recovery.io.raster import (
    read_timeseries,
    _valid_year_str
)


class TestReadTimeseriesDirectoryInput:

    @pytest.fixture
    def filenames(self):
        filenames = ["path/2020.tif", "path/2021.tif", "path/2022.tif"]
        return filenames

    @pytest.fixture
    def rasterio_return(self):
        return xr.DataArray(
                    [[[[0]]]],
                    dims=["band", "time", "y", "x"],
                    attrs={"long_name": ["red"]},
                )
    
    @pytest.mark.parametrize(
        ("tif_paths", "rasterio_return"),
        [
            (
                ["2020.tif"],
                xr.DataArray(
                    [[[[0]]]],
                    dims=["band", "time", "y", "x"],
                    attrs={"long_name": ["red"]},
                ),
            ),
            (
                [
                    "2020.tif",
                    "2021.tif",
                    "2022.tif",
                ],
                xr.DataArray(
                    [[[[0]]]],
                    dims=["band", "time", "y", "x"],
                    attrs={"long_name": ["red"]},
                ),
            ),
            (
                [
                    "2020.tif",
                    "2021.tif",
                    "2022.tif",
                ],
                xr.DataArray(
                    [[[[0]]], [[[0]]], [[[0]]]],
                    dims=["band", "time", "y", "x"],
                    attrs={"long_name": ["red", "blue", "nir"]},
                ),
            ),
            (
                [
                    "2020.tif",
                    "2021.tif",
                    "2022.tif",
                ],
                xr.DataArray(
                    [[[[0, 0], [0, 0]]]],
                    dims=["band", "time", "y", "x"],
                    attrs={"long_name": ["red"]},
                ),
            ),
            (
                [
                    "2020.tif",
                    "2021.tif",
                    "2022.tif",
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
    @patch(
        "spectral_recovery.io.raster._get_tifs_from_dir"
    )
    def test_correct_stacked_output_sizes(
        self,
        mocked_get_tifs,
        mocked_rasterio_open,
        tif_paths,
        rasterio_return,
    ):
        mocked_get_tifs.return_value = tif_paths
        mocked_rasterio_open.return_value = rasterio_return
        stacked_tifs = read_timeseries(
            path_to_tifs="a/dir",
            array_type="numpy",
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
                    [[[[0]]]], dims=["band", "time", "y", "x"], coords={"band": [1]}
                ),
            ),
            (
                [f"tif2020"],
                xr.DataArray(
                    [[[[0]]]], dims=["band", "time", "y", "x"], coords={"band": [1]}
                ),
            ),
            (
                [f"20"],
                xr.DataArray(
                    [[[[0]]]], dims=["band", "time", "y", "x"], coords={"band": [1]}
                ),
            ),
            (
                [f"2020", f"not_a_year", f"2022"],
                xr.DataArray(
                    [[[[0]]]], dims=["band", "time", "y", "x"], coords={"band": [1]}
                ),
            ),
        ],
    )
    @patch(
        "rioxarray.open_rasterio",
    )
    @patch(
        "spectral_recovery.io.raster._get_tifs_from_dir"
    )
    def test_bad_filenames_throws_value_err(
        self,
        mocked_get_tifs,
        mocked_rasterio_open,
        filenames,
        rasterio_return,
    ):
        mocked_get_tifs.return_value = filenames
        mocked_rasterio_open.return_value = rasterio_return
        with pytest.raises(
            ValueError,
        ):
            read_timeseries(
                path_to_tifs="a/dir",
                array_type="numpy",
            )

    @patch(
        "rioxarray.open_rasterio",
    )
    @patch(
        "spectral_recovery.io.raster._get_tifs_from_dir"
    )
    def test_sorted_years(
        self,
        mocked_get_tifs,
        mocked_rasterio_open,
    ):
        sorted_years = [
            np.datetime64("1990"),
            np.datetime64("1992"),
            np.datetime64("2017"),
            np.datetime64("2018"),
        ]
        mocked_get_tifs.return_value = [f"2017.tif", f"2018.tif", f"1992.tif", f"1990.tif"]
        mocked_rasterio_open.return_value = xr.DataArray(
                    [[[[0]]], [[[0]]], [[[0]]]],
                    dims=["band", "time", "y", "x"],
                    coords={"band": [1, 2, 3]},
                    attrs={"long_name": ["blue", "red", "nir"]},
                )
        stacked_tifs = read_timeseries(
            path_to_tifs="a/dir",
            array_type="numpy",
        )
        print(stacked_tifs)
        print(stacked_tifs["time"].data, sorted_years)
        assert np.all(stacked_tifs["time"].data == sorted_years)

    def test_array_type_default_uses_dask_arrays(self):
        stacked_tifs = read_timeseries(
            path_to_tifs="src/tests/test_data/composites/",
            band_names={1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir16", 6: "swir22"}
        )
        assert isinstance(stacked_tifs.data, da.Array)
    
    def test_array_type_numpy_returns_numpy_array(self):
        stacked_tifs = read_timeseries(
            path_to_tifs="src/tests/test_data/composites/",
            band_names={1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir16", 6: "swir22"},
            array_type="numpy",
        )
        assert isinstance(stacked_tifs.data, np.ndarray)

    @patch(
        "rioxarray.open_rasterio",
    )
    @patch(
        "spectral_recovery.io.raster._get_tifs_from_dir"
    )
    def test_int_raster_input_converted_to_float64(
        self,
        mocked_get_tifs,
        mocked_rasterio_open,
        filenames,
        rasterio_return,
    ):
        int_raster = rasterio_return.astype(int)
        mocked_get_tifs.return_value = filenames
        mocked_rasterio_open.return_value = int_raster

        stacked_tifs = read_timeseries(
            path_to_tifs="a/dir",
            array_type="numpy",
        )

        assert np.issubdtype(stacked_tifs.dtype, np.float64)
    
    @patch(
        "rioxarray.open_rasterio",
    )
    @patch(
        "spectral_recovery.io.raster._get_tifs_from_dir"
    )
    def test_float_raster_input_unconverted(
        self,
        mocked_get_tifs,
        mocked_rasterio_open,
        filenames,
        rasterio_return,
    ):
        
        mocked_get_tifs.return_value = filenames
        f32_raster = rasterio_return.astype(np.float32)
        mocked_rasterio_open.return_value = f32_raster
        stacked_tifs = read_timeseries(
            path_to_tifs="a/dir",
            array_type="numpy",
        )
        assert np.issubdtype(stacked_tifs.dtype, np.float32)

        f64_raster = rasterio_return
        mocked_rasterio_open.return_value = f64_raster
        stacked_tifs = read_timeseries(
            path_to_tifs="a/dir",
            array_type="numpy",
        )
        assert np.issubdtype(stacked_tifs.dtype, np.float64)

class TestReadTimeseriesBandNames:
    
    @pytest.fixture
    def filenames(self):
        filenames = ["path/to/2019.tif", "path/to/2020.tif", "path/to/2021.tif"]
        return filenames

    @patch(
        "rioxarray.open_rasterio",
    )
    @patch(
        "spectral_recovery.io.raster._get_tifs_from_dir"
    )
    def test_correct_bands_from_tifs_with_long_name(self, mocked_get_tifs, mocked_rasterio_open, filenames):
        expected_bands = ["B", "R", "N"]
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band": [1, 2, 3]},
            attrs={"long_name": ["blue", "red", "nir"]},
        )
        mocked_get_tifs.return_value = filenames
        mocked_rasterio_open.return_value = rasterio_return

        stacked_tifs = read_timeseries(path_to_tifs="a/dir", array_type="numpy",)
        assert np.all(stacked_tifs["band"].data == expected_bands)

    @patch(
        "rioxarray.open_rasterio",
    )
    @patch(
        "spectral_recovery.io.raster._get_tifs_from_dir"
    )
    def test_correct_bands_from_tifs_w_band_dict(self, mocked_get_tifs, mocked_rasterio_open, filenames):
        expected_bands = ["B", "R", "N"]
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band": [1, 2, 3]},
        )
        mocked_get_tifs.return_value = filenames
        mocked_rasterio_open.return_value = rasterio_return

        stacked_tifs = read_timeseries(
            path_to_tifs="a/dir",
            band_names={1: "blue", 2: "red", 3: "nir"},
            array_type="numpy",
        )

        assert np.all(stacked_tifs["band"].data == expected_bands)

    @patch(
        "rioxarray.open_rasterio",
    )
    @patch(
        "spectral_recovery.io.raster._get_tifs_from_dir"
    )
    def test_invalid_band_name_throws_error(self, mocked_get_tifs, mocked_rasterio_open, filenames):
        rasterio_return = xr.DataArray(
            [[[[0]]]], dims=["band", "time", "y", "x"], coords={"band": [1]}
        )
        mocked_get_tifs.return_value = filenames
        mocked_rasterio_open.return_value = rasterio_return

        with pytest.raises(
            ValueError,
        ):
            read_timeseries(
                path_to_tifs="a/dir",
                band_names={0: "not_a_band"},
                array_type="numpy",
            )

    @patch(
        "rioxarray.open_rasterio",
    )
    @patch(
        "spectral_recovery.io.raster._get_tifs_from_dir"
    )
    def test_band_dict_supersedes_band_desc(self, mocked_get_tifs, mocked_rasterio_open, filenames):
        expected_bands = ["B", "R", "N"]
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band": [1, 2, 3]},
            attrs={"long_name": ["swir", "green", "red"]},
        )
        mocked_get_tifs.return_value = filenames
        mocked_rasterio_open.return_value = rasterio_return

        stacked_tifs = read_timeseries(
            path_to_tifs="a/dir",
            band_names={1: "blue", 2: "red", 3: "nir"},
            array_type="numpy",
        )
        assert np.all(stacked_tifs["band"].data == expected_bands)

    @patch(
        "rioxarray.open_rasterio",
    )
    @patch(
        "spectral_recovery.io.raster._get_tifs_from_dir"
    )
    def test_band_dict_assigns_name_by_key_not_order(self, mocked_get_tifs, mocked_rasterio_open, filenames):
        expected_bands = ["R", "B", "N"]
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band": [1, 2, 3]},
        )
        mocked_get_tifs.return_value = filenames
        mocked_rasterio_open.return_value = rasterio_return

        stacked_tifs = read_timeseries(
            path_to_tifs="a/dir",
            band_names={2: "blue", 1: "red", 3: "nir"},
            array_type="numpy",
        )
        # assert
        print(stacked_tifs["band"].data, expected_bands)
        assert_array_equal(stacked_tifs["band"].data, expected_bands)

    @patch(
        "rioxarray.open_rasterio",
    )
    @patch(
        "spectral_recovery.io.raster._get_tifs_from_dir"
    )
    def test_band_dict_missing_mapping_throws_value_err(self, mocked_get_tifs, mocked_rasterio_open, filenames):
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band": [1, 2, 3]},
        )
        mocked_get_tifs.return_value = filenames
        mocked_rasterio_open.return_value = rasterio_return

        with pytest.raises(
            ValueError,
        ):
            _ = read_timeseries(
                path_to_tifs="a/dir",
                band_names={0: "red", 2: "nir"},
                array_type="numpy",
            )

    @patch(
        "rioxarray.open_rasterio",
    )
    @patch(
        "spectral_recovery.io.raster._get_tifs_from_dir"
    )
    def test_band_dict_invalid_mapping_throws_value_err(self, mocked_get_tifs, mocked_rasterio_open, filenames):
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band": [1, 2, 3]},
        )
        mocked_get_tifs.return_value = filenames
        mocked_rasterio_open.return_value = rasterio_return

        with pytest.raises(
            ValueError,
        ):
            _ = read_timeseries(
                path_to_tifs="a/dir",
                band_names={0: "blue", 1: "red", 2: "nir", 3: "swir"},
                array_type="numpy",
            )

    @patch(
        "rioxarray.open_rasterio",
    )
    @patch(
        "spectral_recovery.io.raster._get_tifs_from_dir"
    )
    def test_no_band_desc_or_band_names_throws_value_err(self, mocked_get_tifs, mocked_rasterio_open, filenames):
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band": [1, 2, 3]},
        )
        mocked_get_tifs.return_value = filenames
        mocked_rasterio_open.return_value = rasterio_return

        with pytest.raises(
            ValueError,
        ):
            _ = read_timeseries(
                path_to_tifs="a/dir",
                array_type="numpy",
            )
    
class TestReadTimeseriesDictInput:

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_dict_of_tifs_is_read(self, rasterio_mock):
        path_dict_int = {2015: "path/to/some_file.tif", 2016: "path/to/another.tif", 2017: "path/to/last.tif"}
        bands = {0: "blue"}
        rasterio_mock.return_value = xr.DataArray(
            [[[0.]]], 
            dims=["band", "y", "x"]
        )
        int_years = read_timeseries(path_to_tifs=path_dict_int, band_names=bands, array_type="numpy")
        expceted_years = [2015, 2016, 2017]
        for i, d in enumerate(int_years.time.values):
            assert pd.to_datetime(d).year == expceted_years[i]

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_dict_of_tifs_maps_years_correctly(self, rasterio_mock):
        out_of_order_tifs = {"2017": "path/to/some_file.tif", "2015": "path/to/another.tif", "2016": "path/to/last.tif"}
        bands = {0: "blue"}
        rasterio_mock.side_effect = [
            xr.DataArray(
            [[[0.]]], 
            dims=["band", "y", "x"]
        ), 
        xr.DataArray(
            [[[1.]]], 
            dims=["band", "y", "x"]
        ), 
        xr.DataArray(
            [[[2.]]], 
            dims=["band", "y", "x"]
        )
        ]
        excepted_output = xr.DataArray(
            [[[[1.]],[[2.]],[[0.]]]],
            dims=["band", "time", "y", "x"],
            coords={"time": [np.datetime64("2015"), np.datetime64("2016"), np.datetime64("2017")], "band": ["B"]}
        )

        output_ts = read_timeseries(path_to_tifs=out_of_order_tifs, band_names=bands, array_type="numpy")
        assert output_ts.equals(excepted_output)

class TestValidYearStr:

    def test_valid_year_returns_true(self):
        _valid_year_str("2017")
        _valid_year_str("1803")
        _valid_year_str("2550")

    def test_bad_format_raises_value_err(self):
        with pytest.raises(ValueError):
            _valid_year_str("17")
            _valid_year_str("11803")
    
    def not_year_raises_value_err(self):
        with pytest.raises(ValueError):
            _valid_year_str("not_year")
            _valid_year_str("Y2018")
