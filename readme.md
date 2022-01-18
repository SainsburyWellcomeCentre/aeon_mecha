# aeon_mecha

Project Aeon's main repository for manipulating acquired data. Includes preprocessing, querying, plotting, and analysis modules.

## Set-up Instructions

### Remote set-up 

#### [On SWC's HPC](docs/env_setup/swc_hpc_setup.md)

### Local set-up

#### Prereqs

- Install [git](https://git-scm.com/downloads)
	- If you are not familiar with git, just confirm the default settings during installation.

- Clone this repository: create a 'ProjectAeon' directory in your home directory, clone this repository there, and `cd` into the cloned directory:
```
mkdir ~/ProjectAeon
cd ~/ProjectAeon
https://github.com/SainsburyWellcomeCentre/aeon_mecha
cd aeon_mecha
```
**Ensure you stay in the `~/ProjectAeon/aeon_mecha` directory for the rest of the set-up instructions, regardless of which set-up procedure you follow below.**

For using this code on your local computer, follow one of the three below procedures for setting up a virtual environment (we recommend the first option, using Anaconda and the conda package manager). All commands below should be run in a bash terminal (Windows users can use the 'mingw64' terminal that comes installed with git). The various set-up tools mentioned below do some combination of python version, environment, package, and package dependency management. For basic information on the differences between these tools, see this [blog post](https://dev.to/bowmanjd/python-tools-for-managing-virtual-environments-3bko#hatch).

#### [Set-up with Anaconda (python distribution) and conda (python version manager, environment manager, package manager, and package dependency manager)](docs/env_setup/anaconda_conda_setup.md)

#### [Set-up with Pyenv (python version manager) and Poetry (python environment manager, package manager, and package dependency manager)](docs/env_setup/pyenv_poetry_setup.md)

#### [Set-up with Pip (python package manager) and Venv (python environment manager)](docs/env_setup/pip_venv_setup.md)

## Repository Contents

## Todos

- add to [repository contents](#repository-contents)
