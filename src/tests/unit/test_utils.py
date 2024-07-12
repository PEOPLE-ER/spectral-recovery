import pytest
import xarray as xr

from spectral_recovery._utils import common_and_long_to_short
from spectral_recovery._utils import maintain_rio_attrs


def test_green_maps_to_G_not_G1():
    names_dict = common_and_long_to_short(["G", "G1"])
    assert names_dict["green"] != "G1"
    assert names_dict["green"] == "G"


def test_rededge_throws_key_error():
    names_dict = common_and_long_to_short(["RE1", "RE2", "RE3"])
    with pytest.raises(KeyError):
        names_dict["rededge"]

def test_xarr_rio_attrs_maintianed():
    test_stack = xr.DataArray([0], dims=["a"]).rio.write_crs("EPSG:4326", inplace=True)

    @maintain_rio_attrs
    def test(stack, a, b, c):
        # create new instance of stack with + operation
        stack_2 = stack + stack
        return stack_2
    
    result = test(test_stack, 1, 2, 3)

    assert result.rio.crs == test_stack.rio.crs
    
def test_any_order_params_successfully_keeps_rio_attrs():

    @maintain_rio_attrs
    def test(stack, a, b, c):
        return stack
    
    @maintain_rio_attrs
    def test2(a, b, stack, c):
        return stack
    
    test_stack = xr.DataArray([0], dims=["a"]).rio.write_crs("EPSG:4326", inplace=True)
    test(a=1, b=2, c=3, stack=test_stack)
    test2(1, 2, test_stack, "3")

def test_more_than_one_data_array_with_diff_crs_throws_val_err():
    test_stack1 = xr.DataArray([0], dims=["a"]).rio.write_crs("EPSG:3857", inplace=True)
    test_stack2 = xr.DataArray([0], dims=["a"]).rio.write_crs("EPSG:4326", inplace=True)
    @maintain_rio_attrs
    def test(stack1, stack2):
        return
    
    with pytest.raises(ValueError):
        test(test_stack1, test_stack2)
    
