# aeon_mecha
![aeon_mecha_env_build_and_tests](https://github.com/SainsburyWellcomeCentre/aeon_mecha/actions/workflows/build_env_run_tests.yml/badge.svg?branch=main)
[![aeon_mecha_tests_code_coverage](https://codecov.io/gh/SainsburyWellcomeCentre/aeon_mecha/branch/main/graph/badge.svg?token=973EC1CG03)](https://codecov.io/gh/SainsburyWellcomeCentre/aeon_mecha)

Project Aeon's main repository for manipulating acquired data. Includes modules for loading raw data, performing quality control on raw data, processing raw data, and ingesting processed data into a DataJoint MySQL database.

## Set-up Instructions

The various set-up tools mentioned below do some combination of python version, environment, package, and package dependency management. For basic information on the differences between these tools, see this [blog post](https://dev.to/bowmanjd/python-tools-for-managing-virtual-environments-3bko#hatch).

### Remote set-up on SWC's HPC

#### Prereqs

1. Ssh into the HPC and clone this repository to your home directory.
```
ssh <your_SWC_username>@ssh.swc.ucl.ac.uk
mkdir ~/ProjectAeon
cd ~/ProjectAeon
git clone https://github.com/SainsburyWellcomeCentre/aeon_mecha
cd aeon_mecha
```

#### Set-up

Ensure you stay in the `~/ProjectAeon/aeon_mecha` directory for the rest of the set-up instructions, regardless of which set-up procedure you follow below.

[Option 1](./docs/env_setup/remote/miniconda_conda_remote_setup.md): **miniconda** (python distribution) and **conda** (python version manager, environment manager, package manager, and package dependency manager)

- *Note*: [mamba](https://mamba.readthedocs.io/en/latest/), a faster alternative to conda, is now installed as a module on the HPC, so the above instructions can be followed using 'mamba' instead of 'conda' if you prefer.

[Option 2](./docs/env_setup/remote/pip_venv_remote_setup.md): **pip** (python package manager) and **venv** (python environment manager)

### Local set-up

#### Prereqs

All commands below should be run in a bash shell (Windows users can use the 'mingw64' terminal that is included when installing git).

1. Clone this repository: create a 'ProjectAeon' directory in your home directory, clone this repository there, and `cd` into the cloned directory:
```
mkdir ~/ProjectAeon
cd ~/ProjectAeon
git clone https://github.com/SainsburyWellcomeCentre/aeon_mecha
cd aeon_mecha
```

#### Set-up

Ensure you stay in the `~/ProjectAeon/aeon_mecha` directory for the rest of the set-up instructions, regardless of which set-up procedure you follow below.

[Option 1](./docs/env_setup/local/miniconda_conda_local_setup.md): **miniconda** (python distribution) and **conda** (python version manager, environment manager, package manager, and package dependency manager)

- *Note*: **mambaforge** and **mamba** can be used as faster, drop-in replacements for 'miniconda' and 'conda', respectively. You can set up the Aeon environment using them, following roughly the same instructions as above. See [here](https://biapol.github.io/blog/mara_lampert/getting_started_with_mambaforge_and_python/readme.html) for more info.

[Option 2](./docs/env_setup/local/pip_venv_local_setup.md): **pip** (python package manager) and **venv** (python environment manager)

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

Sainsbury Wellcome Centre Foraging Behaviour Working Group. (2023). Aeon: An open-source platform to study the neural basis of ethological behaviours over naturalistic timescales,  https://doi.org/10.5281/zenodo.8411157

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8411157.svg)](https://zenodo.org/doi/10.5281/zenodo.8411157)
