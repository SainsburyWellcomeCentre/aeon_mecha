# aeon_mecha
![aeon_mecha_env_build_and_tests](https://github.com/SainsburyWellcomeCentre/aeon_mecha/actions/workflows/build_env_run_tests.yml/badge.svg?branch=main)
[![aeon_mecha_tests_code_coverage](https://codecov.io/gh/SainsburyWellcomeCentre/aeon_mecha/branch/main/graph/badge.svg?token=973EC1CG03)](https://codecov.io/gh/SainsburyWellcomeCentre/aeon_mecha)

Project Aeon's main repository for manipulating acquired data. Includes modules for loading raw data, performing quality control on raw data, processing raw data, and ingesting processed data into a DataJoint MySQL database.

## Set-up Instructions

### Remote set-up on SWC's HPC

1. SSH into the HPC and clone this repository:
```
ssh <your_SWC_username>@ssh.swc.ucl.ac.uk
mkdir ~/ProjectAeon && cd ~/ProjectAeon
git clone https://github.com/SainsburyWellcomeCentre/aeon_mecha
cd aeon_mecha
```
2. Follow the [remote set-up](./docs/env_setup/remote/remote_setup.md) instructions.

### Local set-up

> Run commands in a bash shell. Windows users can use the 'mingw64' terminal included with git.

1. Clone this repository:
```
mkdir ~/ProjectAeon && cd ~/ProjectAeon
git clone https://github.com/SainsburyWellcomeCentre/aeon_mecha
cd aeon_mecha
```
2. Follow the [local set-up](./docs/env_setup/local/local_setup.md) instructions.

## Repository Contents

- `.github/workflows/` : GitHub actions workflows for building the environment and running tests 
- `aeon/` : Source code for the Aeon Python package 
    - `aeon/dj_pipeline`: Source code for the Aeon DataJoint MySQL database pipeline
    - `aeon/io`: Source code for loading raw data
    - `aeon/processing`: Source code for processing raw data
    - `aeon/qc`: Source code for quality control of raw data
    - `aeon/schema`: Examples of 'experiment schemas': variables that can be used to load raw data from particular experiments
- `docker/` : Dockerfiles for building Docker images for the Aeon DataJoint MySQL database pipeline.
- `docs/` : Documentation for the Aeon project
    - `docs/devs/` : Documentation for developers
    - `docs/env_setup/` : Documentation for setting up the Aeon Python environment
    - `docs/examples/` : Aeon usecase examples
    - `docs/using_hpc_jupyterhub.md` : Instructions for using Jupyter notebooks to access Aeon data via SWC's HPC
    - `docs/using_online_dashboard.md` : Instructions for connecting to Aeon's online dashboard
- `env_config/` : Configuration files for the Aeon Python environment
- `tests/` : Unit and integration tests
    - `tests/data` : Data used by tests

## Citation Policy

If you use this software, please cite it as below:

D. Campagner, J. Bhagat, G. Lopes, L. Calcaterra, A. G. Pouget, A. Almeida, T. T. Nguyen, C. H. Lo, T. Ryan, B. Cruz, F. J. Carvalho, Z. Li, A. Erskine, J. Rapela, O. Folsz, M. Marin, J. Ahn, S. Nierwetberg, S. C. Lenzi, J. D. S. Reggiani, SGEN group – SWC GCNU Experimental Neuroethology Group. _Aeon: an open-source platform to study the neural basis of ethological behaviours over naturalistic timescales._ Preprint at https://doi.org/10.1101/2025.07.31.664513 (2025)

[![DOI:10.1101/2025.07.31.664513](https://img.shields.io/badge/DOI-10.1101%2F2025.07.31.664513-AE363B.svg)](https://doi.org/10.1101/2025.07.31.664513)
