# Quick Overview

Here is a simple workflow for computing recovery metrics which demonstrates the core functionalities of the spectral_recovery package. For more detailed examples and in-depth explanations, see the [User Guide](https://PEOPLE-ER.github.io/Spectral-Recovery/terminology).

Begin by importing the spectral_recovery tool:

``` py
import spectral_recovery as sr
```
## Define your Restoration Site

The spectral_recovery tool requires a spatial and temporal definition of your restoration site (i.e where is the site and when did the disturbance and restoration events occur?). 

The read_restoration_polygon function can read any vector format file containing your resoration polygon and will return a geopandas.GeoDataFrame containing your polygon and dates:

``` py 
rest_site = sr.read_restoration_polygon(
    path="my_restoration_poly.gpkg",
    disturbance_start="2005",
    restoration_start="2006",
)
```

## Read in Timeseries Data

The spectral_recovery tool also requires an annual timeseries of rasters as input. If your images are written to disk (TIF), spectral_recovery can help you read your images into a xarray.DataArray object with appropriate dimension labels and spatial attributes. By default, the images are read in lazily using dask.arrays. 

``` py
ts = sr.read_timeseries(
    path_to_tifs="tifs/",
    band_names={1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir16" }
)
```

## Compute Indices

Select a set of indices. The indices you select should represent the characteristics you are interested in computing recovery metrics for.

!!! info

    spectral_recovery uses the modern spectral index catalogue [Awesome Spectral Indices](https://awesome-ee-spectral-indices.readthedocs.io/en/latest/) to support computation of 100+ vegetation and burn indices. Since 100+ choices can be overwhelming, a smaller list of "core" indices, indices that have been proven effective in previous spectral recovery research, is provided [here](https://PEOPLE-ER.github.io/Spectral-Recovery/spectral_recovery_tool/#333-spectral-indices). 

``` py   
indices = sr.compute_indices(ts, indices=["NBR", "GNDVI", "SAVI"])
```

## Compute Recovery Targets

Some recovery metrics require a recovery target which represents the desired spectral characteristics of your resoration site. A novel feature of the `spectral_recovery` package is that it allows users to choose between using a [historic recovery target](https://people-er.github.io/Spectral-Recovery/about/#14-recovery-targets) or a [reference recovery target](https://people-er.github.io/Spectral-Recovery/about/#14-recovery-targets) when computing recovery metrics. Be sure to consult the user guide and current literature when deciding which type of target works best for computing your recovery metrics.

!!! tip

    Only Y2R and R80P require recovery targets. If you are not computing these metrics, you can skip computing a recovery target and not pass anything to the `recovery_target` parameter in `compute_metrics` (default is None).

The spectral_recovery tool provides 2 methods to faciliate recovery target computation. For more information on the recovery target methods that are available with the tool, see [Recovery Targets](http://127.0.0.1:8000/Spectral-Recovery/recovery_targets/).

### Historic Recovery Targets

To compute a recovery target based on the historic conditions of your restoration site:

``` py
hist_targets = sr.targets.median_target(
    polygon="reference_site_polygons.gpkg"
    timeseries_data=indices
    reference_start="2022"
    reference_end="2022"
    scale="polygon"
)
```

### Reference Recovery Targets

To compute a recovery target based conditions in a reference site, you will need to provide a path to vector file or a geopandas.DataFrame containing your reference site polygon(s) alongside the timeseries data and reference years. 

``` py
ref_targets = sr.targets.median_target(
    polygon="reference_sys_polygons.gpkg"
    timeseries_data=indices
    reference_start="2022"
    reference_end="2022"
    scale="polygon"
)
```

## Compute Recovery Metrics

Finally, once you've defined your restration site, computed your indices, and derived your recovery targets, you can compute recovery metrics:

``` py
metrics = sr.compute_metrics(
    metrics=["dNBR", "Y2R", "R80P"],
    restoration_polygons=restoration_site,
    timeseries_data=indices,
    recovery_target=hist_targets
)
```

You can investigate individual metrics by selecting the metric name from the metrics dimension:

``` py
y2r = metric.sel(metrics="Y2R")
```

or you can write metrics to file using the `to_raster` function in the rioxarray extension:

``` py
y2r.rio.to_raster("y2r.tif")
```
