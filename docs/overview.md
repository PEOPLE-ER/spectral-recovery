# Quick Overview

Here is a simple workflow for computing recovery metrics which demonstrates the core functionalities of the spectral_recovery package. For more detailed examples and in-depth explanations, see the User Guide.

Begin by importing the spectral_recovery tool:

```python
import spectral_recovery as sr
```

## Read in Timeseries Data

The spectral_recovery tool requires an annual timeseries of rasters as input. If your images are written to disk (TIF), spectral_recovery can help you read your images into a xarray.DataArray object with appropriate dimension labels and spatial attributes with the [read_timeseries]() function. By default, images are read in lazily using dask.arrays. 

```python
ts = sr.read_timeseries("src", band_names={0: "red", 1: "green", 2:, 3, 4})
```

## Compute Indices

spectral_recovery uses the modern spectral index catalogue [Awesome Spectral Indices](https://awesome-ee-spectral-indices.readthedocs.io/en/latest/) to support computation of 100+ vegetation and burn indices.

```python
# SHOW LIST
```
because 100+ choices can be overwhelming, a list of "core" indices is also provided. These are indices that have been proven effective by previous spectral recovery research for identifying _____. 

```python
# SHOW CORE LIST
```

```python
ts_ind = sr.compute_indices(ts, band_names={0: "red", 1: "green", 2:, 3, 4})
```

## Read in Polygon Data

The spectral_recovery tool also requires a spatial and temporal delineation of the restoration site (i.e where is the site and when did the disturbance and restoration events occur?). You can read in this data from vector files using the (read_restoration_polygon)[] function.

```python
rs = sr.restoration_site("src", disturbance_start_year="", restoration_start_year="")
```

The years (disturbance and restoration start years) can be provided as arguments to the function or can be added beforehand as attributes to the polygon's attribute table. See the API documentation for more information.

## Compute Recovery Targets

Recovery targets are a set of clearly define measurable ecosystem attributes (Hobbs & Norton, 1996) which identify the degree of recovery desired for the degraded area (Gann et al., 2019). In remote sensing, these recovery targets can be compared to the conditions of the restoraiton site to estimate restoration success. The spectral_recovery tool provides built-in support for computing _historic recovery targets_ and _reference recovery targets_.

### Computing Historic Recovery Targets

Historic recovery targets are a set of recovery targets derived from the the historic conditions (i.e pre-disturbance years) of your restoration site. This approach is commonly seen in remote sensing applications. 

```python
# Derive targets using the median pixel values from the 2 years prior to disturbance.

hist_rt = sr.targets.median(
    timeseries_data=ts_ind,
    area=rs,
    reference_start_year="",
    reference_end_year="",
    scale="pixel", # get a unique target for each pixel.
)
```

See the [Recovery Targets]() documentation for more built-in methods for computing recovery targets.

### Recovery Targets from Reference Conditions

Reference recovery targets are recovery targets derived from area(s) within the same landscape of the restoration site which exhibit desirable ecosystem attributes and represent a healthy or stable ecosystem condition (Gann et al., 2019). The use of reference recovery targets aligns with modern ecological restoration frameworks which recognize that the influence of multiple environmental drivers like climate change can make historic conditions impossible or even undesirable to reach.

To compute a reference recovery target, simply pass your reference system area/polygon(s), and set your reference years to the window of the current desired years.

```python
# Derive targets from the most current (2023) conditions of the reference site.
ref_rt = sr.targets.median(
    timeseries_data=ts_ind,
    area="test_data/reference_site.gpkg",
    reference_start_year="2023",
    reference_end_year="2023",
    scale="polygon", # get one value (median) for all pixels in the area.
)
```

The only metod compatible with using reference polygons is `st.targets.median(..., scale="polygon")`. However, if you want to try out a different implementation, custom functions can be provided to the tool. See the [Creating Custom Recovery Target Functions]() page for more information.


## Compute Recovery Metrics

Once your timeseries of index data, recovery targets, and restoration site have been defined, recovery metrics can be computed:

```python
metrics = sr.compute_metrics(
    metrics=["Y2R", "R80P", "DNBR"]
    timeseries_data=ts_ind, 
    restoration_site=rs, 
    recovery_targets=hist_rt,
)
```

See the [Recovery Metrics]() page for the full list of metrics available with the spectral_recovery tool.

## Plot 

Visualizing the spectral trajectory of your restoration site can be helpful at many stages of the workflow.

```python
sr.plot_spectral_trajector(
    timeseries_data=ts_ind,
    restoration_site=)
```
