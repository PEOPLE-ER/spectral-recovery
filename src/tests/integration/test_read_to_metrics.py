import pytest
import spectral_recovery as sr

class TestReadingCanPassToIndices:

    def test_indices_and_single_p_to_metrics_completes(self):
        rest_site = sr.read_restoration_polygons(
            path="src/tests/test_data/composites/test_single_polygon.gpkg",
            dist_rest_years={0: [2002, 2003]}
        )
        timeseries = sr.read_timeseries(
            path_to_tifs={
                2002: "src/tests/test_data/composites/2002.tif",
                2003: "src/tests/test_data/composites/2003.tif",
                2004: "src/tests/test_data/composites/2004.tif",
                2005: "src/tests/test_data/composites/2005.tif",
                2006: "src/tests/test_data/composites/2006.tif",
                2007: "src/tests/test_data/composites/2007.tif",
                2008: "src/tests/test_data/composites/2008.tif",
                2009: "src/tests/test_data/composites/2009.tif",
                2010: "src/tests/test_data/composites/2009.tif",
                2011: "src/tests/test_data/composites/2009.tif"
            },
            band_names={1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir16", 6: "swir22"},
            array_type="numpy",
        )
        indices = sr.compute_indices(timeseries, indices=["NBR", "NDVI", "SAVI"])

        metrics = sr.compute_metrics(
            metrics=["dNBR", "YrYr"],
            restoration_polygons=rest_site,
            timeseries_data=indices,
        )
    
    def test_indices_and_multi_polys_to_metrics_completes(self):
        rest_site = sr.read_restoration_polygons(
            path="src/tests/test_data/composites/test_multiple_polygons.gpkg",
            dist_rest_years={0: [2002, 2003], 1: [2002, 2003], 2: [2004, 2006], 3: [2002, 2004], 4: [2005, 2006]}
        )
        timeseries = sr.read_timeseries(
            path_to_tifs={
                2002: "src/tests/test_data/composites/2002.tif",
                2003: "src/tests/test_data/composites/2003.tif",
                2004: "src/tests/test_data/composites/2004.tif",
                2005: "src/tests/test_data/composites/2005.tif",
                2006: "src/tests/test_data/composites/2006.tif",
                2007: "src/tests/test_data/composites/2007.tif",
                2008: "src/tests/test_data/composites/2008.tif",
                2009: "src/tests/test_data/composites/2009.tif",
                2010: "src/tests/test_data/composites/2009.tif",
                2011: "src/tests/test_data/composites/2009.tif"
            },
            band_names={1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir16", 6: "swir22"},
            array_type="numpy",
        )
        indices = sr.compute_indices(timeseries, indices=["NBR", "NDVI", "SAVI"])

        sr.compute_metrics(
            metrics=["dNBR", "YrYr"],
            restoration_polygons=rest_site,
            timeseries_data=indices
        )
    
    def test_indices_and_multi_polys_w_dates_attr_to_metrics_completes(self):
        rest_site = sr.read_restoration_polygons(
            path="src/tests/test_data/composites/test_multiple_polygons.gpkg",
        )
        timeseries = sr.read_timeseries(
            path_to_tifs={
                2002: "src/tests/test_data/composites/2002.tif",
                2003: "src/tests/test_data/composites/2003.tif",
                2004: "src/tests/test_data/composites/2004.tif",
                2005: "src/tests/test_data/composites/2005.tif",
                2006: "src/tests/test_data/composites/2006.tif",
                2007: "src/tests/test_data/composites/2007.tif",
                2008: "src/tests/test_data/composites/2008.tif",
                2009: "src/tests/test_data/composites/2009.tif",
                2010: "src/tests/test_data/composites/2009.tif",
                2011: "src/tests/test_data/composites/2009.tif"
            },
            band_names={1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir16", 6: "swir22"},
            array_type="numpy",
        )
        indices = sr.compute_indices(timeseries, indices=["NBR", "NDVI", "SAVI"])

        metrics = sr.compute_metrics(
            metrics=["dNBR", "YrYr"],
            restoration_polygons=rest_site,
            timeseries_data=indices,
        )
        print(metrics)