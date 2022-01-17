# Set-up on SWC's HPC

0) ssh into the HPC GW1 node:

```
ssh <your_SWC_username>@ssh.swc.ucl.ac.uk
ssh hpc-gw1
```

1) Navigate to the `aeon_mecha` repository on the `/ceph` partition:

`cd /ceph/aeon/aeon/code/ProjectAeon/aeon_mecha`

2) Add miniconda to your system path and create the `aeon` python env from the `env.yml` file:

```
module load miniconda
conda env create -f env.yml
```

3) Using the virtual environment:

`conda activate aeon_env`: activates the virtual environment; any commands now run within this terminal will take place within the virtual environment.

`conda deactivate aeon_env`: deactivates the virtual environment.

4) (Optional) Add commands to `.profile` to add miniconda as an environment module and Bonsai and its dependencies to your system path on startup:

Copy the `.profile` file in this folder to your home directory on the HPC.

## Developing while on the HPC

After you've finished creating the virtual environment, finalize the set-up by activating the environment and adding this repository to your python path within the environment:
```
conda activate aeon_env
python setup.py develop
```

1) Using Jupyter

2) Using PyCharm

3) Using VSCode