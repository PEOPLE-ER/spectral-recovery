""" Test manual interventions made to the spyndex module. """
import pytest
import spyndex as spx



def test_green1_common_name_maps_to_G1():
    assert spx.bands["G1"].common_name == "green1"

def test_green_common_name_maps_to_G():
    assert spx.bands["G"].common_name == "green"

def test_rededgeX_common_name_maps_to_REX():
    assert spx.bands["RE1"].common_name == "rededge1"
    assert spx.bands["RE2"].common_name == "rededge2"
    assert spx.bands["RE3"].common_name == "rededge3"