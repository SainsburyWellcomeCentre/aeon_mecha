# Local set-up using uv

1. Install uv (if not already installed):
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. Sync the project dependencies (this will also create the virtual environment):
```
uv sync
```
3. Optionally install the development dependencies:
```
uv sync --dev
```
4. Using the virtual environment:
    - `uv run <command>`: runs a command within the project's virtual environment (e.g. `uv run python my_script.py`).
    - The virtual environment is managed automatically by uv; no manual activate/deactivate is needed.
