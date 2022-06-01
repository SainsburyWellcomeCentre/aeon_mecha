# Remote set-up using pyenv and poetry

1) Make sure minconda is not on path, install pyenv, add it to your path, and it's initialization commands to `~/.profile` and `~/.bashrc`, and restart your shell: 
```
module unload miniconda
curl https://pyenv.run | bash
echo 'export PATH="$PATH:$HOME/.pyenv/bin"' >> ~/.profile
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec bash
```

2) Install python 3.9.10 with pyenv, set it as the default python version for this directory, and rehash to ensure the shim for python 3.9.10 has been properly set:
```
pyenv install 3.9.10
pyenv local 3.9.10
pyenv rehash
python -V             # this should return 'Python 3.9.10'
```

3) Install poetry and restart shell:
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
exec bash
```

4) Call poetry to create a new virtual environment and install the code dependencies from the `pyproject.toml` and `poetry.lock` files: 
```
poetry run python --versison  # creates env
poetry install                # installs deps into env
```

5) Using the virtual environment: 

`poetry shell`: creates a new terminal in which the virtual environment is activated: any commands in this terminal will take place within the virtual environment.

`exit`: deactivates the virtual environment and closes the poetry shell.

For more information on Pyenv and Poetry, see [this blog post](https://blog.jayway.com/2019/12/28/pyenv-poetry-saviours-in-the-python-chaos/) and the more detailed [pyenv docs](https://github.com/pyenv/pyenv) and [poetry docs](https://python-poetry.org/docs/).

6) For instructions on developing within the `aeon` environment, see [`developing_on_hpc.md`](./developing_on_hpc.md)