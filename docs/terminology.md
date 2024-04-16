# TERMINOLOGY
The spectral recovery tool uses a wide variety of terminology that can at times be hard to follow, especially when first trying to wrap your head around this tool. To address this we've put together this document, which aims to explain potentially confusing terms.

## TIME-SERIES DATA
When we refer to **time-series data** we refer to a multi-dimensional xarray.DataArray (Years, Bands, X, Y). Each year should represent an image pertaining to each time step of interest. We recommend using yearly composites in order to maximize the tool's performance, but feel free to use carefully selected images instead.

## RESTORATION SITE
When we refer to **restoration site** we refer to the area of interest for the study which is typically defined by a vector format polygon as well as dates of interest (which we call restoration_start and disturbance_start).

## REFERENCE SYSTEM
When we refer to a **reference system** we refer to one or multiple polygons that can be used as spectral baselines against which to compare the restoration site pixels.

## SPECTRAL INDEX
When we refer to a **spectral index** we refer to the spectral combination of bands that you wish to use as the basis of your analysis. Please refer to [spyndex](https://github.com/awesome-spectral-indices/spyndex) for more information pertaining to the indexes available.

## RECOVERY METRICS
When we refer to **recovery metrics** we refer to the specific metrics through which you aim to understand your area of interest. These metrics will be output by the tool in the form of xarray.DataArray's which can then be used in further processing or written out in your preferred raster format.

## RECOVERY TARGET
When we refer to a **recovery target** we refer to the spectral baseline against which you wish to assess spectral recovery. This tool enables the use of a historical target baseline or a per-pixel, windowed, or per polygon approach; as well as a reference target baseline. For more information on these baseline methods refer to the [Recovery Target]() page.

## BANDS
When we refer to **bands** we refer to surface reflectance bands of Sentinel, Landsat or other optical satellite mission.