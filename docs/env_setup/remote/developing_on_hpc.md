In the examples below, replace `<your_username>` with your SWC HPC username.

# Developing while on the HPC

After you've finished creating the virtual `aeon` environment, activate the environment. Then, add this repository to your python path within the environment by opening a bash terminal and running `python setup.py develop` within this repo's root folder.

For using an IDE (e.g. PyCharm, VSCode, Jupyter, etc.) from your local machine, you will need to set up local port forwarding from a specified port on the HPC: 

* First, open a terminal and set up SSH local port forwarding to HPC-GW1:  `ssh -L 9999:hpc-gw1:22 <your_username>@ssh.swc.ucl.ac.uk`

* Then, in a new terminal, SSH with trusted x11 forwarding into the HPC via the forwarded port on localhost: `ssh -Y <your_username>@localhost -p 9999`

* Lastly, set up the remote interpreter in your IDE to use the forwarded port (here 9999) on your localhost, and point to the python installed in the `aeon` environment. These instructions can typically be found in your IDE's online documentation. [Here are instructions for PyCharm Professional](https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html), and [here for VSCode](https://code.visualstudio.com/docs/remote/ssh).
