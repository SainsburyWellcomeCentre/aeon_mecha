# Local set-up using uv

> **Note:** This guide uses `uv` as the package manager. If you encounter older references to pip/venv elsewhere, use the uv equivalents shown here.

1. Install uv (if not already installed):
```
curl -LsSf https://astral.sh/uv/install.sh | sh
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