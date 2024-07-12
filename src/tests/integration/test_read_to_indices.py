import pytest
import spectral_recovery as sr

class TestReadingCanPassToIndices:

    def test_read_from_dict_to_indices_completes(self):
        expected_output_dims = {"band": 3, "time": 3, "y": 97, "x": 118}
        expected_output_indices = ["NBR", "NDVI", "SAVI"]

        timeseries = sr.read_timeseries(
            path_to_tifs={
                2002: "src/tests/test_data/composites/2002.tif",
                2003: "src/tests/test_data/composites/2003.tif",
                2004: "src/tests/test_data/composites/2003.tif"
            },
            band_names={1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir16", 6: "swir22"},
            array_type="numpy",
        ) 
        indices = sr.compute_indices(timeseries, indices=expected_output_indices)
        assert dict(indices.sizes) == expected_output_dims
        for idx in expected_output_indices:
            assert idx in indices.band.values
        
    def test_read_from_dir_to_indices_completes(self):
        expected_output_dims = {"band": 3, "time": 10, "y": 97, "x": 118}
        expected_output_indices = ["NBR", "NDVI", "SAVI"]

        timeseries = sr.read_timeseries(
            path_to_tifs="src/tests/test_data/composites",
            band_names={1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir16", 6: "swir22"},
            array_type="numpy",
        ) 
        indices = sr.compute_indices(timeseries, indices=expected_output_indices)
        assert dict(indices.sizes) == expected_output_dims
        for idx in expected_output_indices:
            assert idx in indices.band.values
