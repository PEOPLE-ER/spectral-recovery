# Installation

spectral-recovery requires Python 3.10+. Please ensure you have a compatible version of [Python installed](https://www.python.org/downloads/) before proceeding.

## Stable Release

spectral-recovery can be installed using [pip](https://pip.pypa.io/en/stable/installation/) (or [uv](https://github.com/astral-sh/uv)):

```bash
pip install spectral-recovery
```

The required dependencies are:

- Xarray
- Rasterio
- Rioxarray
- NumPy
- GeoPandas
- Dask
- Distributed
- Spyndex

## From Source

!!! warning

    Unreleased source code is not considered stable and might even contain undetected bugs. Proceed with caution if using source code in your workflows.

spectral-recovery can be installed from source code directly from the project repository:

```bash
pip install git+https://github.com/PEOPLE-ER/spectral-recovery.git#egg=spectral_recovery
```

or from a local clone of the repository:

```bash
git clone https://github.com/PEOPLE-ER/spectral-recovery.git
cd spectral-recovery
pip install -e .
```