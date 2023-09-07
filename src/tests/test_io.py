import xarray as xr


from mock import patch
from spectral_recovery.io.raster import (
    read_and_stack_tifs,
    metrics_to_tifs,
)


class TestReadAndStackTifs:
    @patch(
        "rioxarray.open_rasterio",
        return_value=xr.DataArray([[[[0]]]], dims=["band", "time", "y", "x"], attrs={"long_name":["red"]}),
    )
    def test_stacked_output_no_mask(self,  mocked_rasterio):
        paths = ["a/fake/path/2020", "another/fake/path/2021"]
        stacked_tifs = read_and_stack_tifs(paths_to_tifs=paths)

        assert(stacked_tifs.sizes["time"] == len(paths))
        # assert(stacked_tifs.sizes["band"] == mocked_rasterio.open_rasterio().sizes["band"])
