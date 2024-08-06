import pytest

import xarray as xr
import numpy as np
import spyndex as spx

from typing import List

from unittest.mock import patch

from spectral_recovery._config import REQ_DIMS
from spectral_recovery.indices import (
    compute_indices,
    INDEX_CONSTANT_DEFAULTS,
    TCW,
    TCG,
    GCI,
)

INDICES = list(spx.indices) + ["GCI", "TCW", "TCG"]
BANDS = list(spx.bands)
CONSTANTS = list(spx.constants)


def bands_from_index(indices: List[str]):
    """Return list of bands used in an index"""
    bands = []
    for index in indices:
        for b in spx.indices[index].bands:
            if b in BANDS and b not in bands:
                bands.append(b)
    return bands


def constants_from_index(indices: List[str]):
    """Return list of constants used in an index"""
    constants = []
    for index in indices:
        for b in spx.indices[index].bands:
            if b in CONSTANTS and b not in constants:
                constants.append(b)
    return constants


def platforms_from_index(indices: List[str]):
    """Return list of platforms compatible with an index"""
    platforms = []
    for index in indices:
        for p in spx.indices[index].platforms:
            if p not in platforms:
                platforms.append(p)
    return platforms


class TestComputeIndices:
    @patch("spyndex.computeIndex")
    def test_correct_kwargs_for_compute_index_call_no_index_constants(
        self, mock_spyndex
    ):
        index = "NDVI"
        bands = bands_from_index([index])
        data = xr.DataArray(
            np.ones((len(bands), 1, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"band": bands},
        )
        expected_params = {b: data.sel(band=b) for b in bands}

        compute_indices(data, [index])

        input_kwargs = mock_spyndex.call_args.kwargs
        assert input_kwargs == {"index": [index], "params": expected_params}

    @patch("spyndex.computeIndex")
    def test_correct_kwargs_for_compute_index_call_with_index_constants(
        self, mock_spyndex
    ):
        index = "EVI"
        bands = bands_from_index([index])
        constants = constants_from_index([index])
        data = xr.DataArray(
            np.ones((len(bands), 1, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"band": bands},
        )
        bands = {b: data.sel(band=b) for b in bands}
        constants = {c: spx.constants[c].default for c in constants}
        expected_params = bands | constants

        compute_indices(data, [index], constants=constants)

        input_kwargs = mock_spyndex.call_args.kwargs
        assert input_kwargs == {"index": [index], "params": expected_params}

    @patch("spyndex.computeIndex")
    def test_no_constants_passed_uses_defaults(self, mock_spyndex):
        index = "SAVI"
        bands = bands_from_index([index])

        defaults_dict = INDEX_CONSTANT_DEFAULTS[index]["defaults"]

        data = xr.DataArray(
            np.ones((len(bands), 1, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"band": bands},
        )

        bands_dict = {b: data.sel(band=b) for b in bands}
        expected_params = bands_dict | defaults_dict

        compute_indices(data, [index])

        input_kwargs = mock_spyndex.call_args.kwargs
        assert input_kwargs == {"index": [index], "params": expected_params}

    def test_default_constants_w_shared_constants_throws_value_err(self):
        index = ["SAVI", "EVI"]  # both use L with diff defaults
        bands = bands_from_index(index)

        data = xr.DataArray(
            np.ones((len(bands), 1, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"band": bands},
        )

        with pytest.raises(ValueError):
            compute_indices(data, index)

    @patch("spyndex.computeIndex")
    def test_given_constants_merged_with_defaults(self, mock_spyndex):
        index = "EVI2"
        bands = bands_from_index([index])

        given_constants = {"L": 0.7}  # missing g

        data = xr.DataArray(
            np.ones((len(bands), 1, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"band": bands},
        )

        bands_dict = {b: data.sel(band=b) for b in bands}
        expected_params = (
            bands_dict
            | given_constants
            | {"g": INDEX_CONSTANT_DEFAULTS[index]["defaults"]["g"]}
        )
        compute_indices(data, [index], constants=given_constants)

        input_kwargs = mock_spyndex.call_args.kwargs
        assert input_kwargs == {"index": [index], "params": expected_params}

    @patch("spyndex.computeIndex")
    def test_missing_constants_and_null_default_throws_value_err(self, mock_spyndex):
        index = "NIRvP"
        bands = bands_from_index([index])

        data = xr.DataArray(
            np.ones((len(bands), 1, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"band": bands},
        )

        with pytest.raises(ValueError):
            compute_indices(data, [index])

    def test_return_is_data_array_obj(self):
        index = "SAVI"
        bands = bands_from_index([index])
        constants = constants_from_index([index])
        constants_dict = {c: spx.constants[c].default for c in constants}
        data = xr.DataArray(
            np.ones((len(bands), 1, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"band": bands},
        )

        result = compute_indices(data, [index], constants=constants_dict)

        assert isinstance(result, xr.DataArray)

    def test_correct_dimensions_and_coords_on_result(self):
        index = ["CIRE", "NDVI", "EVI"]
        bands = bands_from_index(index)
        constants = constants_from_index(index)
        constants_dict = {c: spx.constants[c].default for c in constants}

        data = xr.DataArray(
            np.ones((len(bands), 2, 2, 2)),
            dims=["band", "time", "y", "x"],
            coords={
                "band": bands,
                "time": [0, 1],
                "y": [2, 3],
                "x": [4, 5],
            },
        )
        result = compute_indices(data, index, constants=constants_dict)

        assert result.dims == tuple(REQ_DIMS)
        assert list(result.time.values) == list(data.time.values)
        assert list(result.y.values) == list(data.y.values)
        assert list(result.x.values) == list(data.x.values)
        assert list(result.band.values) == index

    def test_missing_bands_throws_exception(self):
        index = "NDVI"
        bands = bands_from_index([index])
        bands_missing = bands.pop()
        data = xr.DataArray(
            np.ones((len(bands_missing), 1, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"band": bands_missing},
        )

        with pytest.raises(Exception):
            result = compute_indices(data, index)

    @pytest.mark.parametrize(
        "index",
        [
            "NDSI",
            "NDTI",
            "BI",
            "DBI",
            "DPDD",
        ],
    )
    def test_unsupported_domain_index_throws_exception(self, index):
        data = xr.DataArray(
            np.ones((1, 1, 1, 1)),
            dims=["band", "time", "y", "x"],
        )

        # Act and Assert
        with pytest.raises(ValueError):
            result = compute_indices(data, [index])


class TestGCI:
    def test_returns_correct_values(self):
        params_dict_1 = {"N": xr.DataArray([0.2]), "G": xr.DataArray([0.4])}
        expected_1 = xr.DataArray([-0.5])  # (0.2/0.4)-1
        params_dict_2 = {"N": xr.DataArray([0.4]), "G": xr.DataArray([0.2])}
        expected_2 = xr.DataArray([1.0])  # (0.4/0.2)-1

        output_1 = GCI(params_dict=params_dict_1)
        output_2 = GCI(params_dict=params_dict_2)

        xr.testing.assert_equal(output_1, expected_1)
        xr.testing.assert_equal(output_2, expected_2)

    def test_throws_key_error_if_params_missing_bands(self):
        params_dict = {"G": xr.DataArray([0.4])}
        with pytest.raises(KeyError) as keyerr:
            GCI(params_dict=params_dict)
        assert "'N'" in str(keyerr.value)


class TestTCW:
    def test_returns_correct_values(self):
        params_dict = {
            "B": xr.DataArray([0.1]),
            "G": xr.DataArray([0.2]),
            "R": xr.DataArray([0.3]),
            "N": xr.DataArray([0.4]),
            "S1": xr.DataArray([0.5]),
            "S2": xr.DataArray([0.6]),
        }
        expected = xr.DataArray(
            [-0.34004999999999996]
        )  # 0.1511*0.1+0.1973*0.2+0.3283*0.3+0.3407*0.4-0.7117*0.5-0.4559*0.6
        output = TCW(params_dict=params_dict)

        xr.testing.assert_equal(output, expected)

    def test_throws_key_error_if_params_missing_bands(self):
        params_dict = {
            "G": xr.DataArray([0.2]),
            "R": xr.DataArray([0.3]),
            "N": xr.DataArray([0.4]),
            "S1": xr.DataArray([0.5]),
            "S2": xr.DataArray([0.6]),
        }
        with pytest.raises(KeyError) as keyerr:
            TCW(params_dict=params_dict)
        assert "'B'" in str(keyerr.value)


class TestTCG:
    def test_returns_correct_values(self):
        params_dict = {
            "B": xr.DataArray([0.1]),
            "G": xr.DataArray([0.2]),
            "R": xr.DataArray([0.3]),
            "N": xr.DataArray([0.4]),
            "S1": xr.DataArray([0.5]),
            "S2": xr.DataArray([0.6]),
        }
        expected = xr.DataArray(
            [-0.010519999999999974]
        )  # -0.2941*(0.1)-0.243*(0.2)-0.5424*(0.3)+0.7276*(0.4)+0.0713*(0.5)-0.1608*(0.6)

        output = TCG(params_dict=params_dict)

        xr.testing.assert_equal(output, expected)

    def test_throws_key_error_if_params_missing_bands(self):
        params_dict = {
            "G": xr.DataArray([0.2]),
            "R": xr.DataArray([0.3]),
            "N": xr.DataArray([0.4]),
            "S1": xr.DataArray([0.5]),
            "S2": xr.DataArray([0.6]),
        }
        with pytest.raises(KeyError) as keyerr:
            TCG(params_dict=params_dict)
        assert "'B'" in str(keyerr.value)
