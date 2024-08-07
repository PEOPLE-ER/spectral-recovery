from importlib import resources

def bc06_wildfire_restoration_site():
    """Restoration sites for British Columbia (BC) 2006 wildfire"""
    with resources.path("spectral_recovery.data", "bc_2006_wildfire_restoration_site.gpkg") as path:
        return str(path)

def bc06_wildfire_landsat_BAP_timeseries():
    """Timeseries of landat BAPs for British Columbia (BC) 2006 wildfire"""
    with resources.path("spectral_recovery.data", "bc_2006_wildfire_landsat_BAP.zarr") as path:
        return str(path)
