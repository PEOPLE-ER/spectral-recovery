<h1 align="center">spectral-recovery</h1>
<p align="center">:artificial_satellite::evergreen_tree::chart_with_upwards_trend: supporting ecosystem restoration through spectral recovery analysis :chart_with_upwards_trend::evergreen_tree::artificial_satellite:</p>

<div align="center">
  
  ![tests](https://github.com/PEOPLE-ER/spectral-recovery/actions/workflows/tests.yml/badge.svg?branch=main)
  <a href="">[![PyPI version](https://badge.fury.io/py/spectral-recovery.svg)](https://badge.fury.io/py/spectral-recovery)</a>
  <a href="">[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PEOPLE-ER/spectral-recovery/HEAD?labpath=docs%2Fnotebooks%2F)</a>
  <a href="">[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)</a>

</div>

---

Github: [https://github.com/PEOPLE-ER/spectral-recovery/](https://github.com/PEOPLE-ER/spectral-recovery/)

Documentation: [https://people-er.github.io/spectral-recovery/](https://people-er.github.io/spectral-recovery/)

PyPi: [https://pypi.org/project/spectral-recovery/](https://pypi.org/project/spectral-recovery/)

---
:bangbang: The first stable release (1.0.0) of spectral-recovery is scheduled for 2024.08.30!

---
## Overview

`spectral-recovery` is an open-source project and Python package that provides simple, centralized, and reproducible methods for performing [spectral recovery analysis](https://people-er.github.io/spectral-recovery/about/#13-looking-at-recovery-trajectories) to support [Ecosystem Restoration](https://people-er.github.io/spectral-recovery/about/#11-ecosystem-restoration) (ER) efforts in forested ecosystems.

The package provides straight-forward interfaces and supplementary documentation to encourage the use of well-founded remote sensing techniques in ER research and projects. To get started, users provide restoration site locations, the years of disturbance and restoration, and annual composites of spectral data. `spectral-recovery` handles the rest!

See [Quick Start](https://github.com/PEOPLE-ER/spectral-recovery?tab=readme-ov-file#quick-start) or our [interactive notebooks](https://mybinder.org/v2/gh/PEOPLE-ER/spectral-recovery/HEAD?labpath=docs%2Fnotebooks%2F) to dive right in, [(in-progress) tutorials](https://people-er.github.io/spectral-recovery/installation/) for detailed instructions, or the [theoretical basis](https://people-er.github.io/spectral-recovery/theoretical_basis/) for in-depth information. 

## Installation

```bash
pip install spectral-recovery
```

## Quick Start

```python
import spectral_recovery as sr
from spectral_recovery import data

# Read in timeseries data
spectral_ts = sr.read_timeseries(
    path_to_tifs=data.bc06_wildfire_landsat_BAP_timeseries(),
    band_names={1: "blue", 2: "green", 3: "red", 4: "nir", 5: "swir16", 6: "swir22"},
)
# Compute indices
index_ts = sr.compute_indices(
    timeseries_data=spectral_ts,
    indices=["NBR", "NDVI"],
)
# Read in restoration site(s)
rest_site = sr.read_restoration_site(
    path=data.bc06_wildfire_restoration_site(),
    dist_rest_years={0: [2006, 2007]},
)
# Compute recovery target for restoration site
median_hist = sr.recovery_targets.historic.median(
    timeseries_data=index_ts,
    restoration_sites=rest_site,
    reference_start=2003,
    reference_end=2005,
    scale="pixel",
)
# Compute recovery metrics for restoration site!
metrics = sr.compute_metrics(
    metrics=["Y2R", "R80P", "YrYr", "deltaIR", "RRI"],
    timeseries_data=index_ts,
    restoration_sites=rest_site,
    recovery_targets=median_hist,
)
# Inspect recovery metrics for the restoration site (site 0)
# e.g what is the site's mean R80P (porportion of 80% of the recovery target)?:
metrics[0].sel(metric="R80P").mean().compute()
# Or, write results out to a TIF:
metrics[0].sel(metric="Y2R").rio.to_raster("site0_y2r.tif")

```
## Documentation

- View background information, static tutorials, and API references in our [project documentation.](https://people-er.github.io/spectral-recovery/)
- Try out an interactive notebook: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/PEOPLE-ER/spectral-recovery/HEAD?labpath=docs%2Fnotebooks%2F)

## Contributing

- Report bugs, suggest features, and see what others are saying on our [GitHub Issues](https://github.com/PEOPLE-ER/spectral-recovery/issues) page.
- Start discussions about the tool on our [discussion page](https://github.com/PEOPLE-ER/spectral-recovery/discussions).
- Want to contribute code? See our [CONTRIBUTING](https://github.com/PEOPLE-ER/spectral-recovery/blob/main/CONTRIBUTING.md) document for more information.

## How to Cite

Publication in progress. For now, when using this tool in your work we ask that you acknowledge as follows:

"spectral-recovery method developed in the PEOPLE-ER Project, managed by Hatfield Consultants, and financed by the European Space Agency."

## License

Copyright 2023 Hatfield Consultants LLP

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
