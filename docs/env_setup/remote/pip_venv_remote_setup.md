# Remote set-up using uv

> **Note:** This guide uses `uv` as the package manager. If you encounter older references to pip/venv elsewhere, use the uv equivalents shown here.

1. Load the uv module (this provides both `uv` and a managed Python):
```
module load uv
```
2. Create a virtual environment and sync the project dependencies:
```
uv venv
uv sync
```
3. Optionally install the development dependencies:
```
uv sync --extra dev
```
4. Using the virtual environment:
    - `uv run <command>`: runs a command within the project's virtual environment (e.g. `uv run python my_script.py`).
    - The virtual environment is managed automatically by uv; no manual activate/deactivate is needed.
5. (Optional) Add commands to the `.profile` file to load uv as an environment module and Bonsai and its dependencies to your system path on startup (this will be initialized each time you SSH into the HPC).
    - Copy the commands in the `.profile_example` file in this folder to your HPC home directory `.profile` file.
6. For instructions on developing within the `aeon` environment, see [`developing_on_hpc.md`](./developing_on_hpc.md)