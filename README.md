# data_management

Code for managing acquired data. Includes preprocessing, querying, and analysis modules.

## Prereqs

- Install [git][git](https://git-scm.com/downloads)

## Set-up

For using this code on your local computer, we recommend following one of the below procedures for setting up a virtual environment. All commands below should be run in a bash terminal (Windows users can use the 'mingw64' terminal that comes installed with git). 

If using this code on SWC's HPC, ... <@todo>

### Set-up with Anaconda

---

### Set-up with Pyenv and Poetry

It is assumed that you already have Python installed on your computer.

0) If you haven't already, create a 'ProjectAeon' directory in your home directory, and clone this repository there:

```
mkdir ~/ProjectAeon
cd ~/ProjectAeon
git clone https://github.com/ProjectAeon/data-management
```

1) Install pyenv:
	- For MacOS and GNU/Linux: `curl https://pyenv.run | bash`
	- For Windows: Follow the installation instructions in the ['pyenv-win' github repo](https://github.com/pyenv-win/pyenv-win#installation), including the "Finish the installation" section.

2) Install python 3.8.0 with pyenv: 
	```
	pyenv install 3.8.0
	pyenv local 3.8.0
	```

3) Within the `~/ProjectAeon/data-mangement` directory, set python 3.9 to be the default version of python for this project: `pyenv local 3.9.4`

4) Install poetry: `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`

5) Call poetry to create a new virtual environment and install the code dependencies: 

```
poetry env use python3.8.0  # activates env
poetry install              # installs deps into env
```

6) Using the virtual environment: `poetry shell` will create a new terminal in which the virtual environment is activated: any commands in this terminal will take place within the virtual environment. From this virtual environment terminal, `exit` will deactivate the environment and close the terminal.

For more information on Pyenv and Poetry, see [this blog post](https://blog.jayway.com/2019/12/28/pyenv-poetry-saviours-in-the-python-chaos/) and the more detailed [pyenv docs](https://github.com/pyenv/pyenv) and [poetry docs](https://python-poetry.org/docs/).


### Set-up with Pip and Virtualenv

### General usage notes

- If using an IDE (e.g. Pycharm, VSCode, etc...), you will have to look up how to integrate a virtual environment (with whichever set-up option you followed) with the IDE. Usually this process is straightforward, e.g. for PyCharm it involves just setting the cloned repository's path as a project within PyCharm. This information should be available in the docs for the IDE.

## Todos

- switch to python 3.9 when stable release is avialble via pyenv-win
- create 'prereqs' section