# aeon_mecha

Project Aeon's main repository for manipulating acquired data. Includes preprocessing, querying, plotting, and analysis modules.

## Set-up Instructions

The various set-up tools mentioned below do some combination of python version, environment, package, and package dependency management. For basic information on the differences between these tools, see this [blog post](https://dev.to/bowmanjd/python-tools-for-managing-virtual-environments-3bko#hatch).

### Remote set-up on SWC's HPC

#### Prereqs

1. Ssh into the HPC GW1 node and clone this repo to your home directory.

```
ssh <your_SWC_username>@ssh.swc.ucl.ac.uk
ssh hpc-gw1
mkdir ~/ProjectAeon
cd ~/ProjectAeon
git clone https://github.com/SainsburyWellcomeCentre/aeon_mecha
```

#### Set-up

Ensure you stay in the `~/ProjectAeon/aeon_mecha` directory for the rest of the set-up instructions, regardless of which set-up procedure you follow below.

[Option 1](./docs/env_setup/remote/miniconda_conda_remote_setup.md): miniconda (python distribution) and conda (python version manager, environment manager, package manager, and package dependency manager)

[Option 2](./docs/env_setup/remote/pyenv_poetry_remote_setup.md): pyenv (python version manager) and poetry (python environment manager, package manager, and package dependency manager)

[Option 3](./docs/env_setup/remote/pip_venv_remote_setup.md): pip (python package manager) and venv (python environment manager)

### Local set-up

#### Prereqs

1. Install [git](https://git-scm.com/downloads). If you are not familiar with git, just confirm the default settings during installation.

2. Clone this repository: create a 'ProjectAeon' directory in your home directory, clone this repository there, and `cd` into the cloned directory:
```
mkdir ~/ProjectAeon
cd ~/ProjectAeon
https://github.com/SainsburyWellcomeCentre/aeon_mecha
cd aeon_mecha
```

#### Set-up

Ensure you stay in the `~/ProjectAeon/aeon_mecha` directory for the rest of the set-up instructions, regardless of which set-up procedure you follow below. All commands below should be run in a bash terminal (Windows users can use the 'mingw64' terminal that comes installed with git).

[Option 1](./docs/env_setup/local/miniconda_conda_local_setup.md): miniconda (python distribution) and conda (python version manager, environment manager, package manager, and package dependency manager)

[Option 2](./docs/env_setup/local/pyenv_poetry_local_setup.md): pyenv (python version manager) and poetry (python environment manager, package manager, and package dependency manager)

[Option 3](./docs/env_setup/local/pip_venv_local_setup.md): pip (python package manager) and venv (python environment manager)

## Repository Contents

## Todos

- add to [repository contents](#repository-contents)
