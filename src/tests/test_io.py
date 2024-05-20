import pytest
import dask.array as da

import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd

from numpy.testing import assert_array_equal
from unittest.mock import patch
from tests.utils import SAME_XR
from spectral_recovery.io.raster import (
    read_timeseries,
    _metrics_to_tifs,
)

from spectral_recovery.io.polygon import read_restoration_polygons



class TestReadAndStackTifs:

    @pytest.fixture
    def filenames(self):
        filenames = ["path/to/2019.tif", "path/to/2020.tif", "path/to/2021.tif"]
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
        stacked_tifs = read_timeseries(
            path_to_tifs=tif_paths,
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
            read_timeseries(
                path_to_tifs=filenames,
                array_type="numpy",
            )

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_correct_bands_from_tifs_with_long_name(self, mocked_rasterio_open, filenames):
        expected_bands = ["B", "R", "N"]
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band": [1, 2, 3]},
            attrs={"long_name": ["blue", "red", "nir"]},
        )
        mocked_rasterio_open.return_value = rasterio_return

        stacked_tifs = read_timeseries(path_to_tifs=filenames, array_type="numpy",)
        assert np.all(stacked_tifs["band"].data == expected_bands)

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_correct_bands_from_tifs_w_band_dict(self, mocked_rasterio_open, filenames):
        expected_bands = ["B", "R", "N"]
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band": [1, 2, 3]},
        )
        mocked_rasterio_open.return_value = rasterio_return

        stacked_tifs = read_timeseries(
            path_to_tifs=filenames,
            band_names={1: "blue", 2: "red", 3: "nir"},
            array_type="numpy",
        )

        assert np.all(stacked_tifs["band"].data == expected_bands)

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_invalid_band_name_throws_error(self, mocked_rasterio_open, filenames):
        rasterio_return = xr.DataArray(
            [[[[0]]]], dims=["band", "time", "y", "x"], coords={"band": [1]}
        )
        mocked_rasterio_open.return_value = rasterio_return

        with pytest.raises(
            ValueError,
        ):
            stacked_tifs = read_timeseries(
                path_to_tifs=filenames,
                band_names={0: "not_a_band"},
                array_type="numpy",
            )

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_band_dict_supersedes_band_desc(self, mocked_rasterio_open, filenames):
        expected_bands = ["B", "R", "N"]
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band": [1, 2, 3]},
            attrs={"long_name": ["swir", "green", "red"]},
        )
        mocked_rasterio_open.return_value = rasterio_return

        stacked_tifs = read_timeseries(
            path_to_tifs=filenames,
            band_names={1: "blue", 2: "red", 3: "nir"},
            array_type="numpy",
        )
        assert np.all(stacked_tifs["band"].data == expected_bands)

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_band_dict_assigns_name_by_key_not_order(self, mocked_rasterio_open, filenames):
        expected_bands = ["R", "B", "N"]
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band": [1, 2, 3]},
        )
        mocked_rasterio_open.return_value = rasterio_return

        stacked_tifs = read_timeseries(
            path_to_tifs=filenames,
            band_names={2: "blue", 1: "red", 3: "nir"},
            array_type="numpy",
        )
        # assert
        print(stacked_tifs["band"].data, expected_bands)
        assert_array_equal(stacked_tifs["band"].data, expected_bands)

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_band_dict_missing_mapping_throws_value_err(self, mocked_rasterio_open, filenames):
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band": [1, 2, 3]},
        )
        mocked_rasterio_open.return_value = rasterio_return

        with pytest.raises(
            ValueError,
        ):
            _ = read_timeseries(
                path_to_tifs=filenames,
                band_names={0: "red", 2: "nir"},
                array_type="numpy",
            )

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_band_dict_invalid_mapping_throws_value_err(self, mocked_rasterio_open, filenames):
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band": [1, 2, 3]},
        )
        mocked_rasterio_open.return_value = rasterio_return

        with pytest.raises(
            ValueError,
        ):
            _ = read_timeseries(
                path_to_tifs=filenames,
                band_names={0: "blue", 1: "red", 2: "nir", 3: "swir"},
                array_type="numpy",
            )

    @patch(
        "rioxarray.open_rasterio",
    )
    def test_no_band_desc_or_band_names_throws_value_err(self, mocked_rasterio_open, filenames):
        rasterio_return = xr.DataArray(
            [[[[0]]], [[[0]]], [[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band": [1, 2, 3]},
        )
        mocked_rasterio_open.return_value = rasterio_return

        with pytest.raises(
            ValueError,
        ):
            _ = read_timeseries(
                path_to_tifs=filenames,
                array_type="numpy",
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
                    coords={"band": [1, 2, 3]},
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
        stacked_tifs = read_timeseries(
            path_to_tifs=filenames,
            array_type="numpy",
        )
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
    def test_int_raster_input_converted_to_float64(
        self,
        mocked_rasterio_open,
        filenames,
        rasterio_return,
    ):
        int_raster = rasterio_return.astype(int)
        mocked_rasterio_open.return_value = int_raster

        stacked_tifs = read_timeseries(
            path_to_tifs=filenames,
            array_type="numpy",
        )

        assert np.issubdtype(stacked_tifs.dtype, np.float64)
    
    @patch(
        "rioxarray.open_rasterio",
    )
    def test_float_raster_input_unconverted(
        self,
        mocked_rasterio_open,
        filenames,
        rasterio_return,
    ):
        
        f32_raster = rasterio_return.astype(np.float32)
        mocked_rasterio_open.return_value = f32_raster
        stacked_tifs = read_timeseries(
            path_to_tifs=filenames,
            array_type="numpy",
        )
        assert np.issubdtype(stacked_tifs.dtype, np.float32)

        f64_raster = rasterio_return
        mocked_rasterio_open.return_value = f64_raster
        stacked_tifs = read_timeseries(
            path_to_tifs=filenames,
            array_type="numpy",
        )
        assert np.issubdtype(stacked_tifs.dtype, np.float64)



class TestReadRestorationPolygons:
    @patch("geopandas.read_file")
    def test_more_than_one_restoration_polygon_throws_value_err(self, mock_read):
        mock_read.return_value = gpd.GeoDataFrame({
            "dist_start": [2015, 2015],
            "rest_start": [2016, 2016],
            "ref_start": [2012, 2012],
            "ref_end": [2012, 2012],
            "geometry": ["POINT (1 2)", "POINT (2 1)"],
        })
        with pytest.raises(ValueError):
            _ = read_restoration_polygons(path="some_path.gpkg")

    @patch("geopandas.read_file")
    def test_less_than_2_cols_throws_value_err(self, mock_read):
        mock_read.return_value = gpd.GeoDataFrame(
            {"dist_start": [2015], "geometry": ["POINT (1 2)"]}
        )
        with pytest.raises(ValueError):
            _ = read_restoration_polygons(path="some_path.gpkg")

    @patch("geopandas.read_file")
    def test_not_2_or_4_cols_throws_value_err(self, mock_read):
        mock_read.return_value = gpd.GeoDataFrame(
            {"dist_start": [2015], "geometry": ["POINT (1 2)"]}
        )
        with pytest.raises(ValueError):
            _ = read_restoration_polygons(path="some_path.gpkg")
        with pytest.raises(ValueError):
            _ = read_restoration_polygons(path="some_path.gpkg")

    @patch("geopandas.read_file")
    def test_not_int_col_throws_value_err(self, mock_read):
        mock_read.return_value = gpd.GeoDataFrame({
            "dist_start": pd.to_datetime("2015"),
            "rest_start": 2016,
            "geometry": ["POINT (1 2)"],
        })
        with pytest.raises(ValueError):
            _ = read_restoration_polygons(path="some_path.gpkg")

    @patch("geopandas.read_file")
    def test_dist_col_greater_than_rest_col_throws_value_err(self, mock_read):
        mock_read.return_value = gpd.GeoDataFrame({
            "dist_start": 2017,
            "rest_start": 2016,
            "geometry": ["POINT (1 2)"],
        })
        with pytest.raises(ValueError):
            _ = read_restoration_polygons(path="some_path.gpkg", disturbance_start="2003", restoration_start="2002")

    @patch("geopandas.read_file")
    def test_passed_dates_set_in_gdf(self, mock_read):
        mock_read.return_value = gpd.GeoDataFrame({
            "geometry": ["POINT (1 2)"],
        })
        
        all_dates = read_restoration_polygons(
            path="some_path.gpkg",
            disturbance_start="2001",
            restoration_start="2002",
        )
        dist_rest_only_dates = read_restoration_polygons(
            path="some_path.gpkg",
            disturbance_start="2001",
            restoration_start="2002",
        )

        assert "dist_start" in all_dates
        assert "rest_start" in all_dates
        assert "dist_start" in dist_rest_only_dates
        assert "rest_start" in dist_rest_only_dates
    
    @patch("geopandas.read_file")
    def test_passed_dates_overwrite_existing_dates(self, mock_read):
        mock_read.return_value = gpd.GeoDataFrame({
            "dist_start": 2017,
            "rest_start": 2016,
            "geometry": ["POINT (1 2)"],
        })
        
        result = read_restoration_polygons(
            path="some_path.gpkg",
            disturbance_start="2001",
            restoration_start="2002",
        )
        assert result.loc[0, "dist_start"] == "2001"
        assert result.loc[0, "rest_start"] == "2002"