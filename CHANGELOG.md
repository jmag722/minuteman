# CHANGELOG

## [v0.2.0] - 2026-10-24

### Added

- shock shape calculator (CPG)

- equilibrium normal shock solver (minuteman now requires Cantera) with a commandline tool nshock-eq

- equilibrium thermodynamic solvers for speed of sound and isentropic coefficient

- uv lockfile

- Pre-commit check

### Changed

- Dependabot updates weekly

### Removed

- Python 3.10 support

### Fixed

- Use of scipy-stubs found pyrefly issues, corrected

## [v0.1.0] - 2026-09-07

Initial release

### Features

- isentropic flow

- normal shock

- oblique shock

- flow around cone

- Rayleigh flow

- Sod shock tube

- Fanno flow

- command line tools (cshock, nshock, isen, oshock, fanno, rayleigh)
