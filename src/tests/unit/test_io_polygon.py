
import pytest

import geopandas as gpd

from spectral_recovery.io.polygon import read_restoration_polygons

class TestReadPolygon:

    def test_polygons_read_into_gpd(self):
        output = read_restoration_polygons("src/tests/test_data/polygon_inbound_epsg3005.gpkg")
        assert isinstance(output, gpd.GeoDataFrame)
        assert len(output) == 1
    
    def test_multiple_polygons_has_one_row_each_polygon(self):
        output = read_restoration_polygons("src/tests/test_data/polygon_multi_inbound_epsg3005.gpkg")
        assert len(output) == 2

    def test_rest_and_dist_years_read_into_gpd_from_attr(self):
        output = read_restoration_polygons("src/tests/test_data/polygon_inbound_epsg3005.gpkg")
        assert list(output) == ["dist_start", "rest_start", "geometry"]
    
    def test_rest_and_dist_years_read_into_gpd_from_param_over_attr(self):
        output = read_restoration_polygons(
            "src/tests/test_data/polygon_multi_inbound_epsg3005.gpkg",
            dist_rest_years={0: [2004, 2005], 1: [2003, 2006]})
        assert list(output) == ["dist_start", "rest_start", "geometry"]
        assert output.loc[0]["dist_start"] == 2004
        assert output.loc[0]["rest_start"] == 2005
        assert output.loc[1]["dist_start"] == 2003
        assert output.loc[1]["rest_start"] == 2006

    def test_polygon_no_dates_no_years_fails(self):
        with pytest.raises(ValueError):
            _ = read_restoration_polygons(
                "src/tests/test_data/polygon_outbound_epsg3005.gpkg"
            )
    
    def test_years_mapped_to_correct_polygon_index(self):
        output = read_restoration_polygons(
            "src/tests/test_data/polygon_multi_inbound_epsg3005.gpkg",
            dist_rest_years={1: [2004, 2005], 0: [2003, 2006]})
        assert output.loc[0]["dist_start"] == 2003
        assert output.loc[0]["rest_start"] == 2006
        assert output.loc[1]["dist_start"] == 2004
        assert output.loc[1]["rest_start"] == 2005

    
    def test_years_with_invalid_index_throws_value_error(self):
        with pytest.raises(ValueError):
            _ = read_restoration_polygons(
                "src/tests/test_data/polygon_multi_inbound_epsg3005.gpkg",
                dist_rest_years={12: [2004, 2005], 1: [2003, 2006]})
