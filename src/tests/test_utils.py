import pytest

from spectral_recovery._utils import  common_and_long_to_short

def test_green_maps_to_G_not_G1():
    names_dict = common_and_long_to_short(["G", "G1"])
    assert names_dict["green"] != "G1"
    assert names_dict["green"] == "G"

def test_rededge_throws_key_error():
    names_dict = common_and_long_to_short(["RE1", "RE2", "RE3"])
    with pytest.raises(KeyError):
        names_dict["rededge"] 
    