import random
import re
import pytest

import xarray as xr
import numpy as np
import spyndex as spx

from typing import List 

from unittest.mock import Mock, patch, ANY
from tests.utils import SAME_XR

from spectral_recovery._config import REQ_DIMS
from spectral_recovery.enums import Index, Platform, BandCommon
from spectral_recovery.indices import (
    compute_indices,
)

ALL_INDICES = list(spx.indices)
ALL_BANDS = list(spx.bands)
ALL_CONSTANTS = list(spx.bands)
BAND_PATTERN = re.compile(r'\b(?:' + '|'.join(map(re.escape, ALL_BANDS)) + r')\b')
CONSTANTS_PATTERN = re.compile(r'\b(?:' + '|'.join(map(re.escape, ALL_CONSTANTS)) + r')\b')

def bands_from_index(indices: List[str]):
    """ Return list of bands used in an index """
    bands = []
    for index in indices:
        formula = spx.indices[index].formula
        matches = BAND_PATTERN.findall(formula)
        for match in matches:
            if match in ALL_BANDS and match not in bands:
                bands.append(match)
    return bands

def constants_from_index(indices: List[str]):
    """ Return list of constants used in an index """
    constants = []
    for index in indices:
        formula = spx.indices[index].formula
        matches = CONSTANTS_PATTERN.findall(formula)
        for match in matches:
            if match in ALL_CONSTANTS and match not in constants:
                constants.append(match)
    return constants

class TestComputeIndices:
    
    @patch("spyndex.computeIndex")
    def test_correct_kwargs_for_compute_index_call(self, mock_spyndex):
        # Set up
        index = "ARVI"
        bands = bands_from_index([index])
        constants = constants_from_index([index])
        data = xr.DataArray(
            np.ones((len(bands), 1, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"band": bands}
        )
        expected_params = {b: data.sel(band=b) for b in bands} | {c: spx.constants[c].default for c in constants}
        print(expected_params)
        # Act
        compute_indices(data, [index])

        # Assert
        input_kwargs = mock_spyndex.call_args.kwargs 
        assert input_kwargs == {"index": [index], "params": expected_params}
    

    def test_return_is_data_array_obj(self):
        # Set up
        index = random.choice(ALL_INDICES)
        bands = bands_from_index([index])
        data = xr.DataArray(
            np.ones((len(bands), 1, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"band": bands}
        )

        # Act
        result = compute_indices(data, [index])

        # Assert
        assert isinstance(result, xr.DataArray)
    

    def test_correct_dimensions_and_coords_on_data_array(self):
        # Set up
        index = random.choices(ALL_INDICES, k=3)
        bands = bands_from_index(index)
        data = xr.DataArray(
            np.ones((len(bands), 1, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"band": bands}
        )

        # Act
        result = compute_indices(data, index)

        # Assert
        assert result.dims == tuple(REQ_DIMS)
        assert result.band.values == bands

    def test_unsupported_index_throws_value_error(self, all_indices):
        raise NotImplementedError
       
    def test_index_with_unrelated_platform_throws_value_error(self):
        raise NotImplementedError
    
    def test_missing_bands_throws_spyndex_exception(self):
        raise NotImplementedError
    
    def test_missing_bands_throws_exception(self):
        raise NotImplementedError

    def test_bad_index_choice_raises_exception(self):
        with pytest.raises(Exception):
            compute_indices(
                xr.DataArray([[[[0]]], [[[0]]]], dims=["band", "time", "y", "x"]),
                ["not_an_index"],
            )


# class TestRequiresBandsDecorator:
#     def test_valid_bands_runs_without_err(self):
#         @requires_bands([BandCommon.BLUE, BandCommon.RED])
#         def to_be_decorated(stack):
#             return "hello"

#         test_stack = xr.DataArray(
#             [[[[0]]], [[[0]]]],
#             dims=["band", "time", "y", "x"],
#             coords={"band": [BandCommon.BLUE, BandCommon.RED]},
#         )
#         assert to_be_decorated(test_stack) == "hello"

#     def test_bands_not_in_stack_throws_value_err(self):
#         @requires_bands([BandCommon.BLUE, BandCommon.RED])
#         def to_be_decorated(stack):
#             return "hello"

#         test_stack = xr.DataArray(
#             [[[[0]]], [[[0]]]],
#             dims=["band", "time", "y", "x"],
#             coords={"band": [BandCommon.BLUE, BandCommon.NIR]},
#         )
#         with pytest.raises(ValueError):
#             to_be_decorated(test_stack)


# class TestCompatiableWithDecorator:
#     def test_supported_platform_in_stack_runs_without_err(self):
#         @compatible_with([Platform.LANDSAT_OLI, Platform.SENTINEL_2])
#         def to_be_decorated(stack):
#             return "hello"

#         test_stack = xr.DataArray(
#             [0], dims=["time"], attrs={"platform": [Platform.LANDSAT_OLI]}
#         )
#         assert to_be_decorated(test_stack) == "hello"

#     def test_platform_diff_than_decorator_throws_value_err(self):
#         @compatible_with([Platform.LANDSAT_TM, Platform.LANDSAT_ETM])
#         def to_be_decorated(stack):
#             return "hello"

#         test_stack = xr.DataArray(
#             [0], dims=["time"], attrs={"platform": [Platform.SENTINEL_2]}
#         )
#         with pytest.raises(ValueError):
#             to_be_decorated(test_stack)
