import rioxarray
import numpy as np

from shapely import Polygon
import geopandas as gpd

# Load your xarray data
# Replace 'your_file.tif' with the path to your raster file
xds = rioxarray.open_rasterio('docs/test_data/annual_composites/2000.tif')

# Create a shapely Polygon
# Replace the coordinates with your desired polygon vertices
polygon = gpd.read_file("docs/test_data/wildfire_516.gpkg")

# Get the bounding box of the polygon
bbox = polygon.geometry.values.bounds

# Clip the xarray using the buffered bounding box
clipped_xds = xds.rio.clip_box(*bbox)

# Extract x/y coordinates from the clipped xarray
x_coords = clipped_xds['x'].values
y_coords = clipped_xds['y'].values

# Determine the corresponding indices in the original xarray
x_indices = np.searchsorted(xds['x'].values, x_coords)
y_indices = np.searchsorted(xds['y'].values, y_coords)

# Create the buffered indices
buffered_x_indices = np.clip(x_indices, 3, xds.sizes['x'] - 4)
buffered_y_indices = np.clip(y_indices, 3, xds.sizes['y'] - 4)

# Use the buffered indices to extract the desired region from the original xarray
buffered_data = xds[:, buffered_y_indices - 3:buffered_y_indices + 4, buffered_x_indices - 3:buffered_x_indices + 4]
print(buffered_data)
