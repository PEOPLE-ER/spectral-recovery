# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- N/A

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
