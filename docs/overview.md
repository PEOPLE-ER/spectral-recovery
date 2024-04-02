# Quick Overview

Here is a simple workflow for computing recovery metrics which demonstrates the core functionalities of the spectral_recovery package. For more detailed examples and in-depth explanations, see the User Guide.

Begin by importing the spectral_recovery tool:

```python
import spectral_recovery as sr
```

## Read in Timeseries Data

The spectral_recovery tool requires an annual timeseries of rasters as input. If your images are written to disk (TIF), spectral_recovery can help you read your images into a xarray.DataArray object with appropriate dimension labels and spatial attributes. By default, the images are read in lazily using dask.arrays. 

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
ts = sr.compute_indices(ts, band_names={0: "red", 1: "green", 2:, 3, 4})

```

## Read in Polygon Data

The spectral_recovery tool also requires a spatial and temporal delineation of the restoration site (i.e where is the site and when did the disturbance and restoration events occur?). Typically, these 

## Compute Recovery Targets

## Compute Recovery Metrics

## Plot 
