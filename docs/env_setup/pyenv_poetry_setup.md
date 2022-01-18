## Local set-up using pyenv and poetry

1) Install pyenv:
	- For MacOS and GNU/Linux: `curl https://pyenv.run | bash`
	- For Windows: Follow the installation instructions in the ['pyenv-win' github repo](https://github.com/pyenv-win/pyenv-win#installation), including the "Finish the installation" section.

2) Install python 3.10 with pyenv, set it as the default python version for this directory, and rehash to ensure the shim for python 3.10 has been properly set:
```
pyenv install 3.10
pyenv local 3.10
pyenv rehash
```

3) Install poetry:
`curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`

4) Call poetry to create a new virtual environment and install the code dependencies from the `pyproject.toml` and `poetry.lock` files: 
```
poetry env use python3.10   # creates env
poetry install              # installs deps into env
```

5) Using the virtual environment: 

`poetry shell`: creates a new terminal in which the virtual environment is activated: any commands in this terminal will take place within the virtual environment.

`exit`: deactivates the virtual environment and closes the poetry shell.

For more information on Pyenv and Poetry, see [this blog post](https://blog.jayway.com/2019/12/28/pyenv-poetry-saviours-in-the-python-chaos/) and the more detailed [pyenv docs](https://github.com/pyenv/pyenv) and [poetry docs](https://python-poetry.org/docs/).