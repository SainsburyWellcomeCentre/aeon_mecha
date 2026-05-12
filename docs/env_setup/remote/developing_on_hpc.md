In the examples below, replace `<your_username>` with your SWC HPC username.

# Developing while on the HPC

## Accessing a compute node

The HPC gateway (`hpc-gw2`) does **not** have `/ceph/aeon` mounted. To access data on Ceph, you must request a compute node via SLURM:

```
srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 -p cpu --time=04:00:00 --mem=16G --pty bash -i
```

> **Note:** You may see `slurm_get_node_energy` errors when the shell starts. These are cosmetic (caused by the energy accounting plugin) and can be ignored. Press Enter to get a clean prompt.

The session will time out after 4 hours of the allocated time.

## Setting up the environment

1. Load uv and sync the project:
```
module load uv
uv sync
```
2. Install the package in editable mode (if developing):
```
uv pip install -e <path_to_aeon_mecha>
```
3. For using an IDE (e.g. PyCharm, VSCode, Jupyter, etc.), you will need to set up local port forwarding from a specified port on the HPC. These instructions can typically be found in your IDE's online documentation. [Here are instructions for PyCharm Professional](https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html), and [here for VSCode](https://code.visualstudio.com/docs/remote/ssh).
