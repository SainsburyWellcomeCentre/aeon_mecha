# Remote set-up using uv

## Accessing a compute node

The HPC gateway (`hpc-gw2`) does **not** have `/ceph/aeon` mounted. To access data on Ceph, you must request a compute node via SLURM:

```
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 -p cpu --time=04:00:00 --mem=16G --pty bash -i
```

> **Note:** You may see `slurm_get_node_energy` errors when the shell starts. These are cosmetic (caused by the energy accounting plugin) and can be ignored. Press Enter to get a clean prompt.

The session will time out after 4 hours of the allocated time.

## Setting up the environment

1. Load the uv module and sync the project dependencies (this will also create the virtual environment):
```
module load uv
uv sync
```
2. Optionally install the development dependencies:
```
uv sync --dev
```
3. Using the virtual environment:
    - `uv run <command>`: runs a command within the project's virtual environment (e.g. `uv run python my_script.py`).
    - The virtual environment is managed automatically by uv; no manual activate/deactivate is needed.
4. (Optional) Add commands to the `.profile` file to load uv as an environment module and Bonsai and its dependencies to your system path on startup (this will be initialized each time you SSH into the HPC).
    - Copy the commands in the `.profile_example` file in this folder to your HPC home directory `.profile` file (you will have to create this file if it doesn't already exist).

## IDE set-up

For using an IDE (e.g. PyCharm, VSCode, Jupyter, etc.), you will need to set up local port forwarding from a specified port on the HPC. These instructions can typically be found in your IDE's online documentation. [Here are instructions for PyCharm Professional](https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html), and [here for VSCode](https://code.visualstudio.com/docs/remote/ssh).
