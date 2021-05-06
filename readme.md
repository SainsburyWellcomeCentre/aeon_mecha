# aeon

Project Aeon's main repository for manipulating acquired data. Includes preprocessing, querying, and analysis modules.

## Set-up on SWC's HPC

0) ssh into the HPC GW1 node:

```
ssh <your_SWC_username>@ssh.swc.ucl.ac.uk
ssh hpc-gw1
```

1) Navigate to the `aeon` repository on the `/ceph` partition:

`cd /ceph/aeon/aeon/code/ProjectAeon/aeon`

2) Add miniconda to your system path and create the `aeon` python env from the `env.yml` file:

```
module load miniconda
conda env create -f env.yml
```

3) Using the virtual environment:

`conda activate aeon_env`: activates the virtual environment; any commands now run within this terminal will take place within the virtual environment.

`conda deactivate aeon_env`: deactivates the virtual environment.

### Developing while on the HPC

1) Using PyCharm

2) Using Jupyter


## Set-up on a local computer

### Prereqs

- Install [git](https://git-scm.com/downloads)
	- If you are not familiar with git, just confirm the default settings during installation.

For using this code on your local computer, follow one of the three below procedures for setting up a virtual environment (we recommend the first option, using Anaconda and the conda package manager). All commands below should be run in a bash terminal (Windows users can use the 'mingw64' terminal that comes installed with git). The various set-up tools mentioned below do some combination of python version, environment, package, and package dependency management. For basic information on the differences between these tools, see this [blog post](https://dev.to/bowmanjd/python-tools-for-managing-virtual-environments-3bko#hatch).

0) First create a 'ProjectAeon' directory in your home directory, and clone this repository there:

```
mkdir ~/ProjectAeon
cd ~/ProjectAeon
git clone https://github.com/ProjectAeon/aeon
```

**Ensure you stay in the parent `~/ProjectAeon` directory for the rest of the set-up instructions, regardless of which set-up procedure you follow below.**

### Set-up with Anaconda (all-in-one python distribution, version manager, environment manager, package manager, and package dependency manager)

1) Install [Anaconda](https://www.anaconda.com/products/individual)
	- If prompted with an "Install for", select "Just Me" (instead of "All Users")
	- Ensure installation is in your home directory:
		- On Windows: `C:\Users\<your_username>\anaconda3`
		- On GNU/Linux: `/home/anaconda3`
		- On MacOS: `/Users/<your_name>/anaconda3`
	- Ensure you add anaconda3 as a path environment variable (even if it says this option is not recommended)
	- Ensure you do *not* register anaconda3 as the default version of Python.
	- _Note_: These installation settings can always be changed posthoc.

2) Create conda environment and install the code dependencies from the `env.yml` file:
```
conda update conda
conda init
conda env create --file env.yml
```

3) Using the virtual environment:

`conda activate aeon_env`: activates the virtual environment; any commands now run within this terminal will take place within the virtual environment.

`conda deactivate aeon_env`: deactivates the virtual environment.

### Set-up with Pyenv (python version manager) and Poetry (python environment manager, package manager, and package dependency manager)

1) Install pyenv:
	- For MacOS and GNU/Linux: `curl https://pyenv.run | bash`
	- For Windows: Follow the installation instructions in the ['pyenv-win' github repo](https://github.com/pyenv-win/pyenv-win#installation), including the "Finish the installation" section.

2) Install python3.9.4 with pyenv, set it as the default python version for this directory, and rehash to ensure the shim for python3.9.4 has been properly set:
```
pyenv install 3.9.4
pyenv local 3.9.4
pyenv rehash
```

3) Install poetry:
`curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`

4) Call poetry to create a new virtual environment and install the code dependencies from the `pyproject.toml` and `poetry.lock` files: 
```
poetry env use python3.9.4  # creates env
poetry install              # installs deps into env
```

5) Using the virtual environment: 

`poetry shell`: creates a new terminal in which the virtual environment is activated: any commands in this terminal will take place within the virtual environment.

`exit`: deactivates the virtual environment and closes the poetry shell.

For more information on Pyenv and Poetry, see [this blog post](https://blog.jayway.com/2019/12/28/pyenv-poetry-saviours-in-the-python-chaos/) and the more detailed [pyenv docs](https://github.com/pyenv/pyenv) and [poetry docs](https://python-poetry.org/docs/).

### Set-up with Pip (python package manager) and Venv (python environment manager)

It is assumed that you already have Python3.9.4 and Pip installed on your computer.

On Windows:

1) Install virtualenv:
`python -m pip install virtualenv`

2) Create virtual environment:
`python -m venv aeon`

3) Activate the virtual environment and install the code dependencies:
`.\aeon\Scripts\activate`
`python -m pip install -r requirements.txt`

4) Using the virtual environment:
`.\aeon\Scripts\activate` activates the virtual environment.
`deactivate` deactivates the virtual environment.

On MacOS and GNU/Linux:

1) Install virtualenv:
`python3 -m pip install virtualenv`

2) Create virtual environment:
`python3 -m venv aeon`

3) Activate the virtual environment and install the code dependencies:
`source aeon/bin/activate`
`python3 -m pip install -r requirements.txt`

4) Using the virtual environment:
`source aeon/bin/activate` activates the virtual environment.
`deactivate` deactivates the virtual environment.

### Developing locally

- After you've finished creating your virtual environment with one of the three above set-up procedures, finalize your set-up by activating the environment and pip installing this repository as an editable package in the environment:
`conda activate aeon_env`
`pip install --editable ./aeon`

- If using an IDE (e.g. Pycharm, VSCode, etc.), you will have to look up how to integrate the virtual environment (with whichever set-up option you followed) with the IDE. Usually this process is straightforward; information can be found from a web search and/or in the docs for the IDE.

## Repository Contents

## Todos

- add to [repository contents](#repository-contents)
- add to [developing while on the HPC](#developing-while-on-the-hpc)
