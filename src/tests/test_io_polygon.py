import pytest

import pandas as pd
import geopandas as gpd

from unittest.mock import patch
from spectral_recovery.io.polygon import read_restoration_polygons

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