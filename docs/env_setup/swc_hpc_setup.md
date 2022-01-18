## Set-up on SWC's HPC

0) ssh into the HPC GW1 node:

```
ssh <your_SWC_username>@ssh.swc.ucl.ac.uk
ssh hpc-gw1
```

1) Clone the `aeon_mecha` repo to your home directory.

```
mkdir ~/ProjectAeon
cd ~/ProjectAeon
git clone https://github.com/SainsburyWellcomeCentre/aeon_mecha
```

2) Add miniconda to your system path and create the `aeon` python env from the `env.yml` file:

```
module load miniconda
conda env create -f env.yml
```

3) Using the virtual environment:

`conda activate aeon`: activates the virtual environment; any commands now run within this terminal will take place within the virtual environment.

`conda deactivate aeon`: deactivates the virtual environment.

4) (Optional) Add commands to `.profile` to add miniconda as an environment module and Bonsai and its dependencies to your system path on startup:

Copy the `.profile` file in this folder to your home directory on the HPC.

## Developing while on the HPC

After you've finished creating the virtual environment, finalize the set-up by activating the environment and adding this repository to your python path within the environment:
```
conda activate aeon
python setup.py develop
```

For using an IDE (e.g. PyCharm, VSCode, Jupyter, etc.) from your local machine, you will need to set up local port forwarding from a specified port on the HPC: 

* First, open a terminal and set up SSH local port forwarding to HPC-GW1 on ports that are not currently in use (here, just as an example, we use port 9999 for localhost and port 22 for HPC-GW1): 
```
ssh -L 9999:hpc-gw1:22 <your_SWC_username>@ssh.swc.ucl.ac.uk
```

* Then, in a new terminal, SSH (with trusted x11 forwarding) into the HPC via the forwarded port on localhost: 
```
ssh -Y jbhagat@localhost -p 9999
```

* Lastly, set up the remote interpreter in your IDE to use the forwarded port (here 9999) on your localhost, and point to the python in the location in which you installed the `aeon` environment (from the instructions above, this is: `/nfs/nhome/live/<your_SWC_username>/ProjectAeon/.conda/envs/aeon/bin/python`). These instructions can typically be found in your IDE's online documentation. [Here are instructions for PyCharm Professional](https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html), and [here for VSCode](https://code.visualstudio.com/docs/remote/ssh).