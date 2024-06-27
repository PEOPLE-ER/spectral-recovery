import pytest
import spectral_recovery as sr

class TestReadingCanPassToIndices:

    def test_read_from_dict_to_indices_completes(self):
        expected_output_dims = {"band": 6, "time": 3, "y": 4, "x": 4}
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
        
    def test_read_from_dir_to_indices_completes(self):
        expected_output_dims = {"band": 6, "time": 3, "y": 4, "x": 4}
        expected_output_indices = ["NBR", "NDVI", "SAVI"]

        timeseries = sr.read_timeseries(
            path_to_tifs="src/tests/test_data/composites",
            band_names={1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir16", 6: "swir22"},
            array_type="numpy",
        ) 
        indices = sr.compute_indices(timeseries, indices=expected_output_indices)
