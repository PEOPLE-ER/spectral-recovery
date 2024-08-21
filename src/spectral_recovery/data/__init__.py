from importlib import resources

def bc06_wildfire_restoration_site():
    """Restoration sites for British Columbia (BC) 2006 wildfire"""
    with resources.path("spectral_recovery.data", "bc06_wildfire_restoration_site.gpkg") as path:
        return str(path)

def bc06_wildfire_landsat_bap_timeseries():
    """Dict of paths to landsat BAPs for British Columbia (BC) 2006 wildfire, 2002-2024"""
    path_dict = {}
    for year in range(2002, 2025):
        with resources.path("spectral_recovery.data.bc06_wildfire_landsat_bap", f"{year}.tif") as path:
            path_dict[year] = str(path)
    return path_dict
