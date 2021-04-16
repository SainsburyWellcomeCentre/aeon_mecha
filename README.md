# data_management

Code for managing acquired data. Includes preprocessing, querying, and analysis modules.

If using this code on SWC's HPC, ...

For using this code on your local computer, we recommend following one of the three below procedures for setting up a virtual environment. All commands below should be run in a bash terminal (Windows users can use the 'mingw64' terminal that comes installed with git). The set-up tools mentioned below do some combination of python version, environment, package, and package dependency management. For basic information on the differences between these tools, see this [blog post](https://dev.to/bowmanjd/python-tools-for-managing-virtual-environments-3bko#hatch).

## Prereqs

- Install [git](https://git-scm.com/downloads)

## Set-up

0) First create a 'ProjectAeon' directory in your home directory, and clone this repository there:

```
mkdir ~/ProjectAeon
cd ~/ProjectAeon
git clone https://github.com/ProjectAeon/data-management
```

Make sure you stay in this directory regardless of which set-up instructions you follow below.

### Set-up with Anaconda (all-in-one python distribution, version manager, environment manager, package manager, and package dependency manager)

1) Install [Anaconda](https://www.anaconda.com/products/individual)

2) Create conda environment and install the code dependencies from the `env.yml` file:
`conda create --file env.yml`

3) Using the virtual environment:
	- On Windows:
	`conda activate aeon`: activates the virtual environment; any commands now run within this terminal will take place within the virtual environment.
	`conda deactivate aeon`: deactivates the virtual environment.
	- On MacOS and GNU/Linux:
	`conda source activate aeon`
	`conda source deactivate aeon`

### Set-up with Pyenv (python version manager) and Poetry (python environment manager, package manager, and package dependency manager)

1) Install pyenv:
	- For MacOS and GNU/Linux: `curl https://pyenv.run | bash`
	- For Windows: Follow the installation instructions in the ['pyenv-win' github repo](https://github.com/pyenv-win/pyenv-win#installation), including the "Finish the installation" section.

2) Install python3.8.0 with pyenv and set it as the default python version for this directory:
```
pyenv install 3.8.0
pyenv local 3.8.0
```

3) Install poetry:
`curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`

4) Call poetry to create a new virtual environment and install the code dependencies from the `pyproject.toml` and `poetry.lock` files: 
```
poetry env use python3.8.0  # creates env
poetry install              # installs deps into env
```

5) Using the virtual environment: 
`poetry shell`: creates a new terminal in which the virtual environment is activated: any commands in this terminal will take place within the virtual environment.
`exit`: deactivates the virtual environment and closes the poetry shell.

For more information on Pyenv and Poetry, see [this blog post](https://blog.jayway.com/2019/12/28/pyenv-poetry-saviours-in-the-python-chaos/) and the more detailed [pyenv docs](https://github.com/pyenv/pyenv) and [poetry docs](https://python-poetry.org/docs/).

### Set-up with Pip (python package manager) and Venv (python environment manager)

It is assumed that you already have Python3.8.0 and Pip installed on your computer.

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

### General usage notes

- If using an IDE (e.g. Pycharm, VSCode, etc.), you will have to look up how to integrate the virtual environment (with whichever set-up option you followed) with the IDE. Usually this process is straightforward; information can be found with a web search and/or in the docs for the IDE.

## Repository Contents

## Todos

- switch to python 3.9 when stable release is available via pyenv-win
- specify settings for Anaconda on install
- instructions for using on HPC
- add to 'repository contents' section
