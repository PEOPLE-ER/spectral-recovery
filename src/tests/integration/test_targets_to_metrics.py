import pytest
import spectral_recovery as sr

class TestWorkflowWithTargets():

    def test_single_polygon_target_hist_target_can_produce_metrics(self):

        rest_site = sr.read_restoration_polygons(
            path="src/tests/test_data/composites/test_single_polygon.gpkg",
            dist_rest_years={0: [2005, 2006]}
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
                2010: "src/tests/test_data/composites/2010.tif",
                2011: "src/tests/test_data/composites/2011.tif"
            },
            band_names={1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir16", 6: "swir22"},
            array_type="numpy",
        )
        indices = sr.compute_indices(timeseries, indices=["NBR", "NDVI", "SAVI"])
        median_target = sr.targets.historic.median(
            restoration_sites=rest_site, 
            timeseries_data=timeseries,
            reference_years={8: [2002, 2003]},
            scale="pixel"
        )

        metrics = sr.compute_metrics(
            metrics=["dNBR", "YrYr"],
            restoration_polygons=rest_site,
            timeseries_data=indices,
            recovery_target=median_target,
        )

        assert list(metrics.data_vars) == [0]
        assert dict(metrics[8].sizes) == {"metric": 2, "band": 3, "y": 97, "x": 118}

    
    def test_single_polygon_target_ref_target_can_produce_metrics(self):
        pass

    def test_multi_polygon_target_hist_target_can_produce_metrics(self):
        pass

    def test_multi_polygon_target_ref_target_can_produce_metrics(self):
        pass
