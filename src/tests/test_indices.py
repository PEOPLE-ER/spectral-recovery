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

INDICES = list(spx.indices)
BANDS = list(spx.bands)
CONSTANTS = list(spx.constants)

def bands_from_index(indices: List[str]):
    """ Return list of bands used in an index """
    bands = []
    for index in indices:
        for b in spx.indices[index].bands:
            if b in BANDS and b not in bands:
                bands.append(b)
    return bands

def constants_from_index(indices: List[str]):
    """ Return list of constants used in an index """
    constants = []
    for index in indices:
        for b in spx.indices[index].bands:
            if b in CONSTANTS and b not in constants:
                constants.append(b)
    return constants

def any_index():
    return 

class TestComputeIndices:
    
    @patch("spyndex.computeIndex")
    def test_correct_kwargs_for_compute_index_call_no_constants(self, mock_spyndex):
        # Set up
        index = "NDVI"
        bands = bands_from_index([index])
        data = xr.DataArray(
            np.ones((len(bands), 1, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"band": bands}
        )
        expected_params = {b: data.sel(band=b) for b in bands}

        # Act
        compute_indices(data, [index])

        # Assert
        input_kwargs = mock_spyndex.call_args.kwargs 
        assert input_kwargs == {"index": [index], "params": expected_params}
    
    @patch("spyndex.computeIndex")
    def test_correct_kwargs_for_compute_index_call_with_constants(self, mock_spyndex):
        # Set up
        index = "EVI"
        bands = bands_from_index([index])
        constants = constants_from_index([index])
        data = xr.DataArray(
            np.ones((len(bands), 1, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"band": bands}
        )
        bands = {b: data.sel(band=b) for b in bands} 
        constants = {c: spx.constants[c].default for c in constants}
        expected_params = bands | constants

        # Act
        compute_indices(data, [index], **constants)

        # Assert
        input_kwargs = mock_spyndex.call_args.kwargs 
        assert input_kwargs == {"index": [index], "params": expected_params}
    
    def test_return_is_data_array_obj(self):
        # Set up
        index = "DpRVIVV"
        print(index)
        bands = bands_from_index([index])
        constants = constants_from_index([index])
        print(bands, constants)
        constants_dict = {c: spx.constants[c].default for c in constants}
        data = xr.DataArray(
            np.ones((len(bands), 1, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"band": bands}
        )

        # Act
        result = compute_indices(data, [index], **constants_dict)

        # Assert
        assert isinstance(result, xr.DataArray)
    

    def test_correct_dimensions_and_coords_on_result(self):
        # Set up
        index = any_index()
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
            }
        )

        # Act
        result = compute_indices(data, index, **constants_dict)

        # Assert
        assert result.dims == tuple(REQ_DIMS)
        assert list(result.time.values) == list(data.time.values)
        assert list(result.y.values) == list(data.y.values)
        assert list(result.x.values) == list(data.x.values)
        assert list(result.band.values) == index
    
    def test_unsupported_domain_throws_value_error(self):
        # Set up
        index = ["not_and_index"]
        data = xr.DataArray(
            np.ones((1, 1, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={
                "band": ["not_a_band"],
            }
        )

        # Act and Assert
        with pytest.raises(Exception):
            result = compute_indices(data, index)
       
    def test_index_with_unrelated_platform_throws_exception(self):
        raise NotImplementedError
    
    
    def test_missing_bands_throws_exception(self):
        index = "NDVI"
        bands = bands_from_index([index])
        bands_missing = bands.pop()
        data = xr.DataArray(
            np.ones((len(bands_missing), 1, 1, 1)),
            dims=["band", "time", "y", "x"],
            coords={"band": bands_missing}
        )

        # Act and Assert
        with pytest.raises(Exception):
            result = compute_indices(data, index)
    
   @pytest.mark.parametrize(
        ("index"),
        [
            (
              random.choice([i for i in spx.indices if spx.indices[i].application_domain == "snow"])
            ),
            (
              random.choice([i for i in spx.indices if spx.indices[i].application_domain == "water"])
            ),
            (
              random.choice([i for i in spx.indices if spx.indices[i].application_domain == "soil"])
            ),
            (
              random.choice([i for i in spx.indices if spx.indices[i].application_domain == "urban"])
            ),
            (
              random.choice([i for i in spx.indices if spx.indices[i].application_domain == "radar"])
            ),
        ],
    )
    def test_unsupported_index_throws_exception(self):
        # Set up


        data = xr.DataArray(
            np.ones((1, 1, 1, 1)),
            dims=["band", "time", "y", "x"],
        )

        # Act and Assert
        with pytest.raises(ValueError):
            result = compute_indices(data, index)


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
