# Spectral Recovery Tool (pre-release)

## Overview

## Installation

```{bash}
pip install spectral-recovery==0.1.0b1
```

#### From CLI

The CLI for the spectral recovery tool can be accessed using the `specrec` command. Try out the help command to see the different options and arguments you can pass.

```{bash}
specrec --help
```

Below is an example of how you would call the `specrec` tool to get the Y2R and the RRI recovery metrics for your restoration area:

```{bash}
specrec -i NBR -i NDVI path/to/annual/composites/ output/path/ path/to/restoration/polygon.gpkg 2015 path/to/reference/polygon.gpkg 2013 2014 Y2R -p 80 RRI -t 5
```

The above command points to a directory containing a set of annual composite tifs, a restoration polygon whose restoration event began in 2015, and a reference polygon with reference years 2013-2014. The `-i` flags at the start of the command indicate that the recovery metrics should be computed using the NBR and NDVI indices, while the subcommands at the end (`Y2R`, `RRI`, etc.) are the choices of recovery metrics to compute for each index.

#### Within Modules (temp. most of this should be in tutorial, not README)

To use the tool within new or existing modules, first import the relevant modules from the `spectral_recovery` package.

```{python}
import geopandas as gpd
import pandas as pd

from spectral_recovery.restoration import RestorationArea, ReferenceSystem
from spectral_recovery.io import raster
from spectral_recovery.enums import Metric, Index
```

Read in your polygon data, set the dates of your timeseries, restoration event, and reference years.

```{python}
# Read in restoration polygon:
restoration_poly = gpd.read_file("path/to/restoration/polygon.gpkg")

# Read in reference polygons:
# If you want a recovery target based on historic conditions in the
# restoration area then use `reference_poly = restoration_poly`
reference_poly = gpd.read_file("path/to/reference/polygon.gpkg")


# the year of the restoration and disturbance events
restoration_start = pd.to_datetime("2015")
disturbance_start = pd.to_datetime("2014")

# the reference years
reference_years = [pd.to_datetime("2011"), pd.to_datetime("2013")]

# All together, this defines a timeseries from 2010-2022 where a restoration 
# started in 2015 from a disturbance in 2014. A recovery target is derived from the 
# two years prior to the disturbance, 2011-2013.

```
Next get a well-formatted xarray.DataArray using `read_and_stack_tifs`

```{python}
xr_stack = raster.read_and_stack_tifs(path_to_tifs="path_to_your_tifs/")
```

Next, if you want to use spectral indices in your recovery metrics, compute the indices! Your annual composites must contain the required bands for each index (e.g NBR needs the NIR and Red bands), otherwise the computation will fail.

```{python}
indices_to_compute = [Index.nbr, Index.ndvi]
indices_stack = xr_stack.satts.indices(indices_to_compute)
```

Then, create a `RestorationArea` object with the indices, dates, and polygons.

```{python}
ra = RestorationArea(
    restoration_polygon=restoration_poly,
    restoration_start=restoration_start,
    disturbance_start=disturbance_start,
    reference_polygon=restoration_poly,
    reference_system=reference_years,
    composite_stack=indices_stack 
)
```

Finally, generate recovery metrics! Simply call the methods for each 
recovery metric that you want over your restoration area.

```{bash}
Y2R_result = ra.Y2R()
R80P_result = ra.R80P()

# Some metrics can be parameterized, like R80P:
R80P_result_default = ra.R80P()
R80P_result_50 = ra.R80P(percent=50) 

```
Finally, if you want to write your metric outputs, use `metrics_to_tifs` function. Results will be written to the output directory with their metric name (e.g "Y2R.tif", "RRI.tif").

```{python}
metrics_to_tifs(metrics_array=Y2R_result, out_dir="your/output/dir/")
```
### Tests (temp. this should be move to CONTRIBUTING or something similar)

Unit tests can be run with the following command
```{bash}
pytest

```

## How to Cite

When using this tool in your work we ask that you please cite the Spectral_Recovery tool as follows:

"Spectral Recovery method developed in the PEOPLE-ER Project, managed by Hatfield Consultants, and financed by the European Space Agency."

## License

Copyright 2023 Hatfield Consultants LLP

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.