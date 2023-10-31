# Remote set-up using pip and venv

1. Ensure that python >=3.9 and pip are installed (if not, install python >=3.9)
```
python -V          # should return >=3.9
python -m pip -V
```
2. Install virtualenv: `python -m pip install virtualenv`
3. Create virtual environment: `python -m venv aeon`
4. Activate the virtual environment and install the code dependencies:
```
source aeon/bin/activate
python -m pip install -e .
```
5. Optionally install the development dependencies:
```
source aeon/bin/activate
python -m pip install -e .[dev]
```
6. Using the virtual environment:
    - `source aeon/bin/activate` activates the virtual environment; any commands now run within this terminal will take place within the virtual environment.
    - `deactivate` deactivates the virtual environment.
7. (Optional) Add commands to the `.profile` file to add miniconda as an environment module and Bonsai and its dependencies to your system path on startup (this will be initialized each time you SSH into the HPC).
    - Copy the commands in the `.profile_example` file in this folder to your HPC home directory `.profile` file.
8. For instructions on developing within the `aeon` environment, see [`developing_on_hpc.md`](./developing_on_hpc.md)