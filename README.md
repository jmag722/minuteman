# Minuteman
[![CI](https://github.com/jmag722/minuteman/actions/workflows/ci.yml/badge.svg)](https://github.com/jmag722/minuteman/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Minuteman is a toolkit for rapid solutions to common engineering problems. I'm currently prioritizing thermo-, gas and fluid dynamics, but will grow to include heat transfer, chemistry, plasma, and others (solid mechanics?).

## Motivation

There are GUI-based and web browser alternatives ([AerospaceWeb](https://aerospaceweb.org/design/scripts/) or [VTT's Compressible Aerodynamics Calculator](https://devenport.aoe.vt.edu/aoe3114/calc.html)) for computing engineering relations like the ones developed here. This repository is for those who:

- want a Python-based, FOSS tool (rather than MATLAB's Aerospace Toolbox)

- want to perform large trade studies, varying different parameters with multiple values (not feasible in a GUI or browser)

- are not comfortable entering their data in a web browser (CUI or otherwise sensitive)

## Installation

Simply pip-installing the package in place (using traditional pip or uv) will work.

```bash
git clone git@github.com:jmag722/minuteman.git
cd minuteman
uv venv
source .venv/bin/activate
uv pip install .
```

## Documentation

Documentation available at [https://jmag722.github.io/minuteman/](https://jmag722.github.io/minuteman/)

## License

Minuteman uses the Apache 2.0 license, see the [license](LICENSE) file for more detail.
