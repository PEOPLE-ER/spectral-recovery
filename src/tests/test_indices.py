import xarray as xr

import pytest
from unittest.mock import Mock
from unittest.mock import patch
from tests.utils import SAME_XR

from spectral_recovery.enums import Index, Platform,BandCommon
from spectral_recovery.indices import (
    compute_indices,
    _indices_map,
    requires_bands,
    compatible_with,
    ndvi, 
    nbr,
    evi,
    avi,
    savi,
    ndwi, 
    tcg, 
    tcw, 
    tcb, 
    sr, 
    ndmi, 
    gci, 
    ndii, 
)

def test_ndvi():
    input_xr = xr.DataArray(
        [[[[2, 1]]],[[[1, 2]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [BandCommon.RED, BandCommon.NIR]},
        attrs={"platform": [Platform.LANDSAT_OLI]}
    )
    expected = xr.DataArray(
        [[[[-0.3333333333333333, 0.3333333333333333]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [Index.NDVI]}
    )
    assert (ndvi(input_xr) == expected).all()

def test_nbr():
    input_xr = xr.DataArray(
        [[[[2, 1]]],[[[1, 2]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [BandCommon.SWIR2, BandCommon.NIR]},
        attrs={"platform": [Platform.LANDSAT_OLI]}
    )
    expected = xr.DataArray(
        [[[[-0.3333333333333333, 0.3333333333333333]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [Index.NBR]}
    )
    assert (nbr(input_xr) == expected).all()

def test_evi():
    input_xr = xr.DataArray(
        [[[[3]]],[[[1]]],[[[2]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [BandCommon.BLUE, BandCommon.RED, BandCommon.NIR]},
        attrs={"platform": [Platform.LANDSAT_OLI]}
    )
    expected = xr.DataArray(
        [[[[-0.18518518518518517]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [Index.EVI]}
    )
    assert (evi(input_xr) == expected).all()

def test_avi():
    input_xr = xr.DataArray(
        [[[[3]]],[[[2]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [BandCommon.RED, BandCommon.NIR]},
        attrs={"platform": [Platform.LANDSAT_OLI]}
    )
    expected = xr.DataArray(
        [[[[1.5874010519681994]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [Index.AVI]}
    )
    assert (avi(input_xr) == expected).all()

def test_savi():
    input_xr = xr.DataArray(
        [[[[1]]],[[[2]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [BandCommon.RED, BandCommon.NIR]},
        attrs={"platform": [Platform.LANDSAT_OLI]}
    )
    expected = xr.DataArray(
        [[[[0.42857142857142855]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [Index.SAVI]}
    )
    assert (savi(input_xr) == expected).all()

def test_ndwi():
    input_xr = xr.DataArray(
        [[[[2, 1]]],[[[1, 2]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [BandCommon.GREEN, BandCommon.NIR]},
        attrs={"platform": [Platform.LANDSAT_OLI]}
    )
    expected = xr.DataArray(
        [[[[0.3333333333333333, -0.3333333333333333]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [Index.NDWI]}
    )
    assert (ndwi(input_xr) == expected).all()

def test_tcg():
    raise NotImplementedError

def test_tcw():
    raise NotImplementedError

def test_tcb():
    raise NotImplementedError

def test_sr():
    input_xr = xr.DataArray(
        [[[[2]]],[[[1]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [BandCommon.RED, BandCommon.NIR]},
        attrs={"platform": [Platform.LANDSAT_OLI]}
    )
    expected = xr.DataArray(
        [[[[0.5]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [Index.SR]}
    )
    assert (sr(input_xr) == expected).all()

def test_ndmi():
    input_xr = xr.DataArray(
        [[[[1, 2]]],[[[2, 1]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [BandCommon.NIR, BandCommon.SWIR1]},
        attrs={"platform": [Platform.LANDSAT_OLI]}
    )
    expected = xr.DataArray(
        [[[[-0.3333333333333333, 0.3333333333333333]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [Index.NDMI]}
    )
    assert (ndmi(input_xr) == expected).all()

def test_gci():
    input_xr = xr.DataArray(
        [[[[2]]], [[[1]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [BandCommon.GREEN, BandCommon.NIR]},
        attrs={"platform": [Platform.LANDSAT_OLI]}
    )
    expected = xr.DataArray(
        [[[[-0.5]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [Index.GCI]}
    )
    assert (gci(input_xr) == expected).all()

def test_ndii():
    input_xr = xr.DataArray(
        [[[[1, 2]]],[[[2, 1]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [BandCommon.NIR, BandCommon.SWIR1]},
        attrs={"platform": [Platform.LANDSAT_OLI]}
    )
    expected = xr.DataArray(
        [[[[-0.3333333333333333, 0.3333333333333333]]]],
        dims=["band", "time", "y", "x"],
        coords={"band": [Index.NDII]}
    )
    assert (ndii(input_xr) == expected).all()

class TestComputeIndices:

    def test_correct_call_from_index_input_to_function(self):
        mock_nbr = Mock(return_value=xr.DataArray([0], dims=["time"]))
        mock_ndvi = Mock(return_value=xr.DataArray([0], dims=["time"]))
        with patch.dict(_indices_map, {Index.NBR: mock_nbr, Index.NDVI: mock_ndvi}):
            input_xr = xr.DataArray(
                    [[[[0]]],[[[0]]]],
                    dims=["band", "time", "y", "x"],
                    )
            compute_indices(
                input_xr,
                ["NBR", "NDVI"],
            )

            mock_nbr.assert_called_with(
                SAME_XR(input_xr)
            )
            mock_ndvi.assert_called_with(
                SAME_XR(input_xr)
            )

    def test_output_has_correct_band_dimension(self):
        """
        The band coordinates should be the Enum equivalent to values passed
        in to the `indices` parameter, in the order they were passed in, 
        regardless of the coordinate values that are returned by the index
        functions. 

        """
        mock_nbr = Mock(return_value=xr.DataArray([[0]], dims=["time", "band"], coords={"band": ["band_1"]}))
        mock_ndvi = Mock(return_value=xr.DataArray([[0]], dims=["time", "band"], coords={"band": ["band_2"]}))
        # Patch the map of index names to functions, replace the functions with mocks
        with patch.dict(_indices_map, {Index.NBR: mock_nbr, Index.NDVI: mock_ndvi}):
            input_xr = xr.DataArray(
                    [[[[0]]],[[[0]]]],
                    dims=["band", "time", "y", "x"]
            )
            res = compute_indices(
                input_xr,
                ["NBR", "NDVI"],
            )
            assert (res["band"].values == [Index.NBR, Index.NDVI]).all()
            assert res.dims == ("time", "band")
    
    def test_bad_index_choice_raises_value_err(self):
        with pytest.raises(
            ValueError):
            compute_indices(
                xr.DataArray(
                    [[[[0]]],[[[0]]]],
                    dims=["band", "time", "y", "x"]
                ),
                ["not_an_index"],
            )

class TestRequiresBandsDecorator:

    def test_valid_bands_runs_without_err(self):
        @requires_bands([BandCommon.BLUE, BandCommon.RED])
        def to_be_decorated(stack):
            return "hello"
        
        test_stack = xr.DataArray(
            [[[[0]]],[[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band": [BandCommon.BLUE, BandCommon.RED]}
        )
        assert to_be_decorated(test_stack) == "hello"
    
    def test_bands_not_in_stack_throws_value_err(self):
        @requires_bands([BandCommon.BLUE, BandCommon.RED])
        def to_be_decorated(stack):
            return "hello"
        
        test_stack = xr.DataArray(
            [[[[0]]],[[[0]]]],
            dims=["band", "time", "y", "x"],
            coords={"band": [BandCommon.BLUE, BandCommon.NIR]}
        )
        with pytest.raises(ValueError):
            to_be_decorated(test_stack)

class TestCompatiableWithDecorator:

    def test_supported_platform_in_stack_runs_without_err(self):
        @compatible_with([Platform.LANDSAT_OLI, Platform.SENTINEL_2])
        def to_be_decorated(stack):
            return "hello"

        test_stack = xr.DataArray([0], dims=["time"], attrs={"platform": [Platform.LANDSAT_OLI]})
        assert to_be_decorated(test_stack) == "hello"
    
    def test_platform_diff_than_decorator_throws_value_err(self):
        @compatible_with([Platform.LANDSAT_TM, Platform.LANDSAT_ETM])
        def to_be_decorated(stack):
            return "hello"

        test_stack = xr.DataArray([0], dims=["time"], attrs={"platform": [Platform.SENTINEL_2]})
        with pytest.raises(ValueError):
            to_be_decorated(test_stack)