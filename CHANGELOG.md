# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- N/A

## [1.0.1] - 2024-08-29

### Added

- New examples TIFs for "basic workflow" notebook (#159)

### Changed

- Update Installation documentation (#159)
- Update Overview documentation (#159)
- Update notebooks (#159)

### Removed

- Delete old example TIFs for "basic workflow" notebook (#159)

### Fixed

- Fix typo in REAME "Quick Start" section (#159)

## [1.0.0] - 2024-08-29

### Added

- Add support for processing multiple restoration sites at once (#136)
- Add GCI, TCW, and TCG indexes (#138)
- Add support for index inputs to read_timeseries (#139)
- Add badge for tests status to README (#150)
- Add citation file to repository (#135)
- Add support for non-continuous timeseries inputs to read_timeseries (#129) 
- Allow users to pass dict of TIF paths to read_timeseries (#130)
- Cast int inputs to float automatically in read_timeseries (#128)
- Add "quick start" section to README (#146)
- Add demo/example data accessible through data module (#143)

### Changed

- Change compute_metrics return from Xarray to dict of Xarray with restoration site ID as keys (#148)
- Refactor recovery metrics to accept only necessary arguments (#148)
- Change recovery targets return from Xarray dict of Xarray with restoration site ID as keys
- Recovery target module refactored into targets.historic and targets.reference sub-modules (#136)
- Update docstrings and API documentation (#149)
- Silence expected runtime warnings from Dask in metric computation (#147)

## Removed

- Remove support for CLI (#137)

### Fixed

- Fix broken spectral-recovery links in documentation (#142)



## [0.4.1] - 2024-04-16

### Fixed

- Triage bad package data import (#123)

## [0.4.0] - 2024-04-16

### Added

- Add restoration site year/date parameters (#98)
- Add notebooks for Binder (#111)
- Add/use default index values where applicable (#99)

### Changed

- Change documentation theme to Mkdocs material (#111)
- Remove recovery target computation from metric and plotting functions (#112)
- Make Dask default array (#96)
- Refactor MedianTarget into median_target function (#112)
- Refactor WindowedTarget into window_target function (#112)

### Fixed

- Fix argument order bug from maintain_rio_attrs wrapper (#114)

## [0.3.3] - 2024-04-05

### Fixed

- Recover from bad/incorrect release (v0.3.2).

## [0.3.2] - 2024-04-04

### Changed

- Remove compute_indices from metric/plotting, add to public API (#102)
- Update documtation to reflect compute_indices in API (#102)

### Fixed

- Fix incorrect reference_polygons is None check (#104) 


## [0.3.1] - 2024-03-19

### Added

- Add initial WindowedTarget recovery target function (#84)
- Re-introduce RRI as available metric (#93)

### Changed

- Change "unrecovered" Y2R values to -9999 (#56)
- Move compute_metrics from restoration to metrics (#78)
- Move plotting logic from restoraiton to plotting (#78)

### Removed

- Metric methods from RestorationArea object (#78)

### Fixed

- Fix non-zero GDF row error (#83)
- Fix dask_exp import err (#82)


## [0.3.0b0] - 2024-02-21

### Added

- Add compute_metrics function (#72).
- Add plot_spectral_trajectory function (#72).
- Add read_restoration_polygon and read_reference_polygon function (#72).

### Changed

- Rename read_and_stack_tifs to read_timeseries (#72).
- Read dates from restoration and reference polygon attribute tables (#70).
- Speed up plotting (#66).
- Alter plotting visulizations for more clarity (#66).
- Updated tutorial (#70).
- Bring CLI up-to-date (#74).

### Removed

- Remove compute_indices, RestorationArea from top-level module (#72).
- Remove use of platform param in io and indices (#73)

## [0.2.3b0] - 2024-02-02

### Added

- Add recovey_target_method parameter to RestorationArea (#61).

### Changed

- Refactor RestorationArea init method (#61).
- Rename recovery_target module to target (#61).
- Updated docstrings and notebooks.
- Refactor median target method into parameterized, callable class (#61).

### Removed

- Remove _ReferenceSystem class (#61).


## [0.2.2b2] - 2024-01-11

### Added

- Add historic vs. reference target details in overview (#47)
- Add corner case tests to Y2R (#42)
- Fix typos in docstrings.

### Fixed

- Fix incorrect Y2R values bug (#42)


## [0.2.1b2] - 2024-01-01

### Added

- Support for Sentinel-2 imagery (#21).
- Add optional param for naming bands (#29).
- Add method for plotting RecoveryArea spectral trajectories (#27)(#30).
- Support for Red Edge and Coastal Aerosol bands (#32).
- New Landsat annual composite test dataset with proper scaling and cropped to a smaller area.
- Support for computation using NumPy arrays (#33).

### Changed

- Update use guide to reflect new API usage.
- Update user guide to show users how to write results using rioxarray.
- Allow close-to-zero differences between image stack and restoration polygons (#37).
- Change recovery target computation from mean to median (#26).
- Allow string inputs instead of datetime objects (#31).
- Lint the codebase (#36).

### Removed

- Temporarily remove tassel-cap indices ("TCW", "TCB", "TCG") (#37).
- Remove old Landsat annual compsite test dataset used in documentation.

### Fixed

- Fix incorrect NDII equation (#37).
- Fix incorrect multi-dimensional median target (#26)(#28).
