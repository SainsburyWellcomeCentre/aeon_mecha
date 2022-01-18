When updating any aeon_env package version, update the following top-level env config files as necessary:
    * `env.yml`
    * `pyproject.toml`
    * `requirements.txt`
    * `.python-version`

Then, in this repo's root directory, run `python -m pip install -U poetry` to upgrade poetry and then run `poetry update` to update the `poetry.lock` file from the `pyproject.toml` file.