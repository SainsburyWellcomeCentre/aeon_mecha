# Remote set-up using miniconda and conda

1) Add miniconda to your system path and create the `aeon` python env from the `env.yml` file:

```
module load miniconda
conda env create -f env.yml
```

2) Using the virtual environment:

`conda activate aeon`: activates the virtual environment; any commands now run within this terminal will take place within the virtual environment.

`conda deactivate aeon`: deactivates the virtual environment.

3) (Optional) Add commands to `.profile` to add miniconda as an environment module and Bonsai and its dependencies to your system path on startup:

Copy the commands in the `.profile` file in this folder to your HPC home directory `.profile` file.

4) For instructions on developing within the `aeon` environment, see [`developing_on_hpc.md`](./developing_on_hpc.md)