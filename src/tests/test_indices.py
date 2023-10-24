import xarray as xr

import pytest
from unittest.mock import Mock
from unittest.mock import patch
from tests.utils import SAME_XR

from spectral_recovery.enums import Index, Platform,BandCommon
from spectral_recovery.indices import (
    compute_indices,
    _indices_map,
)


class TestComputeIndices:

    def test_correct_call_from_index_input_to_function(self):
        mock_nbr = Mock(return_value=xr.DataArray([0], dims=["time"]))
        mock_ndvi = Mock(return_value=xr.DataArray([0], dims=["time"]))
        with patch.dict(_indices_map, {Index.nbr: mock_nbr, Index.ndvi: mock_ndvi}):
            input_xr = xr.DataArray(
                    [[[[0]]],[[[0]]]],
                    dims=["band", "time", "y", "x"] 
                    )
            compute_indices(
                input_xr,
                [Index.nbr, Index.ndvi],
                Platform.landsat,
            )

            mock_nbr.assert_called_with(
                SAME_XR(input_xr)
            )
            mock_ndvi.assert_called_with(
                SAME_XR(input_xr)
            )

    def test_platform_param_attached_to_image_stack_before_call(self):
        mock_nbr = Mock(return_value=xr.DataArray([0], dims=["time"]))
        with patch.dict(_indices_map, {Index.nbr: mock_nbr}):
            input_xr = xr.DataArray(
                    [[[[0]]],[[[0]]]],
                    dims=["band", "time", "y", "x"]
            )
            compute_indices(
                input_xr,
                [Index.nbr],
                Platform.landsat,
            )

            for call_obj in mock_nbr.call_args_list:
                args, kwargs = call_obj
                assert args[0].attrs["platform"] == Platform.landsat

    def test_output_has_correct_band_dimension(self):
        """
        The band dimension coordinates should be the same as the values passed
        in to the `indices` parameter, in the order they were passed in, 
        regardless of the coordinate values that are returned by the index functions.
        """
        mock_nbr = Mock(return_value=xr.DataArray([[0]], dims=["time", "band"], coords={"band": ["band_1"]}))
        mock_ndvi = Mock(return_value=xr.DataArray([[0]], dims=["time", "band"], coords={"band": ["band_2"]}))
        with patch.dict(_indices_map, {Index.nbr: mock_nbr, Index.ndvi: mock_ndvi}):
            input_xr = xr.DataArray(
                    [[[[0]]],[[[0]]]],
                    dims=["band", "time", "y", "x"]
            )
            res = compute_indices(
                input_xr,
                [Index.nbr, Index.ndvi],
                Platform.landsat,
            )
            assert (res["band"].values == [Index.nbr, Index.ndvi]).all()
            assert res.dims == ("time", "band")
    
    def test_bad_index_choice_raises_value_err(self):
        with pytest.raises(
            ValueError,
            match=r"(not a valid index)"):
            compute_indices(
                xr.DataArray(
                    [[[[0]]],[[[0]]]],
                    dims=["band", "time", "y", "x"]
                ),
                [Index.nbr, "not_an_index"],
                Platform.landsat,
            )