# Remote set-up using uv

> **Note:** This guide uses `uv` as the package manager. If you encounter older references to conda or pip elsewhere, use the uv equivalents shown here.

1. Load the uv module and sync the project dependencies:
```
module load uv
uv sync
```
2. Optionally install development dependencies:
```
uv sync --extra dev
```
3. Using the virtual environment:
    - `uv run <command>`: runs a command within the project's virtual environment (e.g. `uv run python my_script.py`).
    - The virtual environment is managed automatically by uv; no manual activate/deactivate is needed.
4. (Optional) Add commands to the `.profile` file to load uv as an environment module and Bonsai and its dependencies to your system path on startup (this will be initialized each time you SSH into the HPC).    
    - Copy the commands in the `.profile_example` file in this folder to your HPC home directory `.profile` file (you will have to create this file if it doesn't already exist).
5. For instructions on developing within the `aeon` environment, see [`developing_on_hpc.md`](./developing_on_hpc.md)