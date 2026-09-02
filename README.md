# MinuteMAN

*Mechanical & Aerospace eNgineering in a Minute!*

<!-- --8<-- [start:badges] -->

[![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fjmag722%2Fminuteman%2Frefs%2Fheads%2Fmaster%2Fpyproject.toml&style=flat&logo=python&logoColor=yellow&label=Python)](https://github.com/jmag722/minuteman/blob/master/pyproject.toml)
[![CI](https://github.com/jmag722/minuteman/actions/workflows/ci.yml/badge.svg)](https://github.com/jmag722/minuteman/actions/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with pyrefly](https://img.shields.io/endpoint?url=https://pyrefly.org/badge.json)](https://github.com/facebook/pyrefly)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<!-- --8<-- [end:badges] -->

<!-- --8<-- [start:intro] -->

MinuteMAN is a toolkit for rapid solutions to engineering problems.

##### Current features include:

- thermodynamics

- compressible flow lookup tables (isentropic flow, normal shocks, Rayleigh flow, Fanno flow)

- oblique shock solver

- Sod shock tube solver

- cone flow solver (Taylor-Maccoll equations)

<!-- --8<-- [end:intro] -->

<!-- --8<-- [start:motivation] -->

## Motivation

There are GUI-based and web browser alternatives to solving aerospace or compressible flow problems - resources such as ([AerospaceWeb](https://aerospaceweb.org/design/scripts/) or [VTT's Compressible Aerodynamics Calculator](https://devenport.aoe.vt.edu/aoe3114/calc.html)) are great for a quick calculation. MATLAB supplies its own toolbox for many of these calculations. This repository is for those who:

- want a Python-based, FOSS tool that can be integrated directly into their own custom workflow

- want to perform large trade studies, varying different parameters with multiple values (not feasible in a GUI or browser)

- are not comfortable entering their CUI or otherwise sensitive data in a web browser

<!-- --8<-- [end:motivation] -->

## Installation

This package is installable using pip or uv.

```bash
git clone git@github.com:jmag722/minuteman.git
cd minuteman
uv venv
source .venv/bin/activate
uv pip install .
```

## Documentation

Documentation available at [https://jmag722.github.io/minuteman/](https://jmag722.github.io/minuteman/)
<!-- --8<-- [start:roadmap] -->

## Roadmap

- equilibrium solvers (shocks, isentropic flow, cone flow)

- shock standoff predictions

- 1D nozzle calculations

- mixed Rayleigh-Fanno flow for a varying area duct

- nozzle design with  method-of-characteristics

- engines

<!-- --8<-- [end:roadmap] -->

## License

Minuteman uses the Apache 2.0 license, see the [license](LICENSE) file for more detail.
