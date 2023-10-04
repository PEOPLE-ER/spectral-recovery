# Spectral Recovery Tool 

### Installation and Environment

First, clone the repository

```{bash}
git clone https://github.com/PEOPLE-ER/Vegetation-Recovery-Assessment.git
cd Vegetation-Recovery-Assessment/
```

#### Installing from Wheel

Build the package and install wheel.

```{bash}
python -m build
pip install dist/spectral_recovery-0.1-py3-none-any.whl
```
The `spectral_recovery` package is now available through import statements and a CLI. See the "Using the Spectral Recovery Tool" notebook for example usage.

#### Installing for Development Mode

To download and perform/test development tasks within the project, enter development mode using an "editable install".

```{bash}
python -m venv .venv
pip install --editable .
```

Now the `spectral_recovery` package is accesible as if it was installed in `.venv` (WIP: see "Using the Spectral Recovery Tool" notebook for usage examples). This installation lets you quickly develop the package without building new distributions. 

When done with a development task, you can simply uninstall the package as you normally would using pip, `pip uninstall spectral_recovery`.

### Running

#### From CLI

The CLI for the spectral recovery tool can be accessed using the `specrec` command. Run `specrec --help` for information about the parameters. Below is an example of a run,

```{bash}
specrec -i NBR -i NDVI path_to_your_tifs/ output/path/ path_to_your_restoration/polygon.gpkg 2015 path_to_your/reference_polygon.gpkg 2013 2014 Y2R -p 80 RI -t 5
```

The above command points to a directory ("path_to_your_tifs/") of annumal tifs, a restoration polygon ("path_to_your_restoration/polygon.gpkg") that experienced disturbance in 2015, and a reference polygon that recovery targets should be derived from the years 2013-2014. The run will compute NBR and NDVI, and for each index will compute the Y2R and RI recovery metrics. One tif for each metrics will be written to "output/path/".

#### Within Modules

To use the tool within new or exisitng modules, first import the relevant modules from the `spectral_recovery` package.

```{python}
import geopandas as gpd
import pandas as from

from spectral_recovery.restoration import RestorationArea, ReferenceSystem
from spectral_recovery.io import raster
from spectral_recovery.enums import Metric, Index, BandCommon
```

Read in your polygon data with geopandas, set the dates of your timeseries, restoration event, and reference years.

```{python}
# Define years:
# the years your TIFs cover
start_year, end_year = [pd.to_datetime("2010"), pd.to_datetime("2022")]

# the year of the restoration event
restoration_year = pd.to_datetime("2015")

# the years to derive reference/recovery target conditions from
reference_years = [pd.to_datetime("2013"), pd.to_datetime("2014")]

# All together, this defines a timeseries from 2010-2022 where a restoration 
# event occured in 2015, and a recovery target can be derived from the 
# two years prior to the disturbance, 2013-2014.

# Read in restoration polygon:
restoration_poly = gpd.read_file("path/to/restoration/polygon.gpkg")

# Read in reference polygons:
# If you want a recovery target based on historic conditions in the
# restoration area then use `reference_poly = restoration_poly`
reference_poly = gpd.read_file("path/to/referene/polygon.gpkg")

```
Next get a well-formated xarray.DataArray using `read_and_stack_tifs`

```{python}
xr_stack = raster.read_and_stack_tifs(
    path_to_tifs="path_to_your_tifs/",
    path_to_mask=None,
    )
```

Check that the stack will be accepted in the `spectral_recovery` tool using the custom the `satts` (short for: satellite timeseries) accessor.

```{python}
print(timeseries.satts.valid)
```

If the stack is valid, you can then compute the indices. Please ensure you have the required bands for each index in your TIF, otherwise the computation will fail.

```{python}
indices_to_compute = [Index.NBR, Index.NDVI]
indices = stack.satts.indices(indices_to_compute)
```
Then, initialize a `RestorationArea` object with the indices, dates, and restoration polygon, and compute metrics using the `metrics()` method.

```{python}
metrics = [Metric.Y2R, Metric.RI]
metrics_array = RestorationArea(
            restoration_polygon=restoration_poly,
            restoration_year=restoration_year,
            reference_polygon=restoration_poly,
            reference_system=reference_years,
            composite_stack=indices,
        ).metrics(metrics)
```
Finally, if you want to write the metric output, use `metrics_to_tifs`

```{python}
metrics_to_tifs(
            metrics_array=metrics_array,
            out_dir="some_output_directory/",
        )
```
### Tests

Unit tests can be run with the following command

```{bash}
pytest

```
