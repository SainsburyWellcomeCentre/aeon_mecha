# Remote set-up using miniconda and conda

1. Add miniconda to your system path and create the `aeon` python env from the `env.yml` file:
```
module load miniconda
conda env create -f env_config/env.yml
```
2. Optionally install development dependencies:
```
conda activate aeon
conda env update -f env_config/env_dev.yml
```
3. Using the virtual environment:
    - `conda activate aeon`: activates the virtual environment; any commands now run within this terminal will take place within the virtual environment.
    - `conda deactivate aeon`: deactivates the virtual environment.
4. (Optional) Add commands to the `.profile` file to add miniconda as an environment module and Bonsai and its dependencies to your system path on startup (this will be initialized each time you SSH into the HPC).    
    - Copy the commands in the `.profile_example` file in this folder to your HPC home directory `.profile` file.
5. For instructions on developing within the `aeon` environment, see [`developing_on_hpc.md`](./developing_on_hpc.md)