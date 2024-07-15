import pytest
import spectral_recovery as sr


class TestWorkflowWithTargets:

    def test_single_polygon_target_hist_target_can_produce_metrics(self):

        rest_site = sr.read_restoration_polygons(
            path="src/tests/test_data/composites/test_single_polygon.gpkg",
            dist_rest_years={0: [2005, 2006]},
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
                2011: "src/tests/test_data/composites/2011.tif",
            },
            band_names={
                1: "blue",
                2: "green",
                3: "red",
                4: "nir",
                5: "swir16",
                6: "swir22",
            },
            array_type="numpy",
        )
        indices = sr.compute_indices(timeseries, indices=["NBR", "NDVI", "SAVI"])

        median_target = sr.targets.historic.median(
            restoration_sites=rest_site,
            timeseries_data=indices,
            reference_years={0: [2002, 2003]},
            scale="pixel",
        )
        metrics = sr.compute_metrics(
            metrics=["dNBR", "YrYr"],
            restoration_polygons=rest_site,
            timeseries_data=indices,
            recovery_target=median_target,
        )

        assert list(metrics.data_vars) == [0]
        assert dict(metrics[0].sizes) == {"metric": 2, "band": 3, "y": 61, "x": 45}

    def test_single_polygon_target_ref_target_can_produce_metrics(self):
        rest_site = sr.read_restoration_polygons(
            path="src/tests/test_data/composites/test_single_polygon.gpkg",
            dist_rest_years={0: [2005, 2006]},
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
                2011: "src/tests/test_data/composites/2011.tif",
            },
            band_names={
                1: "blue",
                2: "green",
                3: "red",
                4: "nir",
                5: "swir16",
                6: "swir22",
            },
            array_type="numpy",
        )
        indices = sr.compute_indices(timeseries, indices=["NBR", "NDVI", "SAVI"])

        ref_target = sr.targets.reference.median(
            reference_sites="src/tests/test_data/composites/test_reference.gpkg",
            timeseries_data=indices,
            reference_start=2011,
            reference_end=2011,
        )
        metrics = sr.compute_metrics(
            metrics=["dNBR", "YrYr"],
            restoration_polygons=rest_site,
            timeseries_data=indices,
            recovery_target=ref_target,
        )

        assert list(metrics.data_vars) == [0]
        assert dict(metrics[0].sizes) == {"metric": 2, "band": 3, "y": 61, "x": 45}

    def test_multi_polygon_target_hist_target_can_produce_metrics(self):
        rest_site = sr.read_restoration_polygons(
            path="src/tests/test_data/composites/test_multiple_polygons.gpkg",
            dist_rest_years={
                0: [2005, 2006],
                1: [2004, 2005],
                2: [2004, 2006],
                3: [2005, 2006],
                4: [2003, 2004],
            },
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
                2011: "src/tests/test_data/composites/2011.tif",
            },
            band_names={
                1: "blue",
                2: "green",
                3: "red",
                4: "nir",
                5: "swir16",
                6: "swir22",
            },
            array_type="numpy",
        )
        indices = sr.compute_indices(timeseries, indices=["NBR", "NDVI", "SAVI"])

        hist_target = sr.targets.historic.median(
            restoration_sites=rest_site,
            timeseries_data=indices,
            reference_years={
                0: [2002, 2003],
                1: [2002, 2004],
                2: [2003, 2004],
                3: [2004, 2004],
                4: [2002, 2002],
            },
            scale="pixel",
        )
        metrics = sr.compute_metrics(
            metrics=["Y2R", "R80P"],
            restoration_polygons=rest_site,
            timeseries_data=indices,
            recovery_target=hist_target,
        )
        assert list(metrics.data_vars) == [0, 1, 2, 3, 4]
        assert dict(metrics[0].sizes) == {"metric": 2, "band": 3, "y": 86, "x": 115}

    def test_multi_polygon_target_ref_target_can_produce_metrics(self):
        rest_site = sr.read_restoration_polygons(
            path="src/tests/test_data/composites/test_multiple_polygons.gpkg",
            dist_rest_years={
                0: [2005, 2006],
                1: [2004, 2005],
                2: [2004, 2006],
                3: [2005, 2006],
                4: [2003, 2004],
            },
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
                2011: "src/tests/test_data/composites/2011.tif",
            },
            band_names={
                1: "blue",
                2: "green",
                3: "red",
                4: "nir",
                5: "swir16",
                6: "swir22",
            },
            array_type="numpy",
        )
        indices = sr.compute_indices(timeseries, indices=["NBR", "NDVI", "SAVI"])
        ref_target = sr.targets.reference.median(
            reference_sites="src/tests/test_data/composites/test_reference.gpkg",
            timeseries_data=indices,
            reference_start=2011,
            reference_end=2011,
        )
        metrics = sr.compute_metrics(
            metrics=["dNBR", "YrYr"],
            restoration_polygons=rest_site,
            timeseries_data=indices,
            recovery_target=ref_target,
        )
        assert list(metrics.data_vars) == [0, 1, 2, 3, 4]
        assert dict(metrics[0].sizes) == {"metric": 2, "band": 3, "y": 86, "x": 115}
