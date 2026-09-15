# MinuteMAN

*Mechanical & Aerospace eNgineering at a Minute's notice!*

<!-- --8<-- [start:badges] -->

[![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fjmag722%2Fminuteman%2Frefs%2Fheads%2Fmaster%2Fpyproject.toml&style=flat&logo=python&logoColor=yellow&label=Python)](https://github.com/jmag722/minuteman/blob/master/pyproject.toml)
[![CI](https://github.com/jmag722/minuteman/actions/workflows/ci.yml/badge.svg)](https://github.com/jmag722/minuteman/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/jmag722/minuteman/graph/badge.svg?token=0IW4BPRRC9)](https://codecov.io/github/jmag722/minuteman)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with pyrefly](https://img.shields.io/endpoint?url=https://pyrefly.org/badge.json)](https://github.com/facebook/pyrefly)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<!-- --8<-- [end:badges] -->

<!-- --8<-- [start:intro] -->

MinuteMAN is a toolkit for rapid solutions to mechanical and aerospace engineering problems. It prioritizes algebraic and ordinary differential equation methods for use in everyday engineering decision-making.

<!-- --8<-- [end:intro] -->

<!-- --8<-- [start:roadmap] -->

## Current Features & Roadmap

Current and upcoming features include (in no particular order):

- [x] Thermodynamics
- [ ] Isentropic flow
    - [x]  CPG
    - [ ] Eq.
- [ ] 1D nozzle calculations
- [ ] Normal shocks
    - [x] CPG
    - [ ] Eq.
- [ ] Oblique shocks
    - [x] CPG
    - [ ] Eq.
- [ ] Conical shocks (Taylor-Maccoll equations)
    - [x] CPG
    - [ ] Eq.
- [x] Rayleigh flow
- [x] Fanno flow
- [ ] Mixed Rayleigh-Fanno flow with varying area
- [x] Sod shock tube solver
- [x] Shock shape predictions for blunt bodies
    - [x] calorically perfect gas (CPG)
    - [ ] equilibrium gas (Eq.)
- [ ] 2D Nozzle design via method-of-characteristics
- [ ] Gas turbine engines

<!-- --8<-- [end:roadmap] -->

## Installation

This package is installable using pip or uv.

```bash
git clone git@github.com:jmag722/minuteman.git
cd minuteman
uv sync
```

## Documentation

Documentation available at [https://jmag722.github.io/minuteman/](https://jmag722.github.io/minuteman/)

<!-- --8<-- [start:motivation] -->

## Motivation

I've always been amazed by the mathematical prowess of early aerospace engineers. One only needs to skim a few NASA papers from the 1950s-70s to see often laborious derivations leading the reader from intractable PDEs to a closed-form solution. When compute power was limited, they developed an astute intuition for when and how to make critical simplifying assumptions.

Such classical methods may appear of limited use in modern aerodynamics, where the go-to approach is to simply overrefine the mesh, aggressively turn on all the physics, and crank it through STAR-CCM or ANSYS. Modern numerical methods have come far and solved very complicated problems.

But I think these "old-school" methods can still be highly valuable, as they can:

- serve as a quick "back-of-the-envelope" estimate for the real solution
- bound the trade space
- give insight to help reduce the number of iterations needed to create a good CFD mesh (where's the shock, cell height, etc)
- be used for verification of CFD when there is an absence of experimental data

Often at work, I've found myself wishing for such a reliable, open-source implementation of these methods. There is no shortage of GUI-based and web browser alternatives for solving aerospace or compressible flow problems - resources such as ([AerospaceWeb](https://aerospaceweb.org/design/scripts/) or [VTT's Compressible Aerodynamics Calculator](https://devenport.aoe.vt.edu/aoe3114/calc.html)) are great for a quick calculation. MATLAB supplies its own toolbox for many of these calculations.

**This repository is for those who**:

- want a Python-based, FOSS tool *that can be integrated directly into their custom workflow* (and is non-GPL licensed)

- need to verify a CFD model in the absence of experimental data

- desire help preparing and even automating a CFD mesh to account for the estimated solution (shocks, cell sizes, etc)

- want to *perform large trade studies, varying different parameters with multiple values* (not feasible in a GUI or browser)

- are not comfortable entering their *CUI or otherwise sensitive data* in an LLM or web browser

<!-- --8<-- [end:motivation] -->


## License

Minuteman uses the Apache 2.0 license, see the [license](LICENSE) file for more detail.
