# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2021-01-27

### Added

- `PauliNet`:
    - Mean-field Jastrow and backflow
- H2O and H4 rectangle systems
- Support for Python 3.9

### Changed

- `PauliNet`/`OmniSchNet`:
    - API

### Removed

- `PauliNet`:
    - Separate Jastrow and backflow factories
    - `mo_factory`, functionality replaced with a mean-field backflow
    - Real-space backflow

### Fixed

- Systems with n_up/n_down = 0

## [0.2.0] - 2020-08-19

### Added

- Command-line interface

### Changed

- `PauliNet`:
    - Nuclear and electronic cusp corrections on by default
    - `omni_kwargs` accepted by `PauliNet()` instead of `from_hf()`
    - All keyword arguments to `from_hf()` are passed to `PauliNet()`
- `Sampler`:
    - `from_mf()` changed to `from_wf()`, doesn't use PySCF object by default

## [0.1.1] - 2020-07-24

Rerelease of 0.1.0 with added package metadata.

## [0.1.0] - 2020-07-24

This is the first official release of DeepQMC.

At this moment, DeepQMC should be still considered a research code.

### Added

- Core functionality to run variational quantum Monte Carlo with Pytorch
- PauliNet, a deep neural network ansatz

[unreleased]: https://github.com/deepqmc/deepqmc/compare/0.3.0...HEAD
[0.3.0]: https://github.com/deepqmc/deepqmc/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/deepqmc/deepqmc/compare/0.1.1...0.2.0
[0.1.1]: https://github.com/deepqmc/deepqmc/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/deepqmc/deepqmc/releases/tag/0.1.0
