import spectral_recovery as sr
from spectral_recovery.enums import Platform, Index
import geopandas as gpd

stack = sr.read_and_stack_tifs("annual_composites/test_baps/", [Platform.landsat_tm, Platform.landsat_etm, Platform.landast_oli])
indices = sr.compute_indices(stack, indices=[Index.ndvi, Index.gndvi])
rest_polys = gpd.read_file('test_restoration_polygon.gpkg')
ra = sr.RestorationArea( restoration_polygon=rest_polys, restoration_start="2015", reference_years=["2014", "2015"], composite_stack=indices, )
ra.plot_spectral_timeseries()