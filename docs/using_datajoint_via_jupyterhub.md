
The examples below use the SWC HPC username `jbhagat`. When you run the commands, replace this username with your username.

## Connecting to jupyterhub server:

1) In a terminal, set up SSH local port forwarding to HPC-GW1 on a port that is not currently in use:
    ```
    ssh -L 9999:hpc-gw1:22 jbhagat@ssh.swc.ucl.ac.uk
    ```

2) In a new terminal, SSH into the HPC via the forwarded port on localhost, activate the aeon environment, and open the jupyterhub server on the HPC via its IP on a specified port:
    ```
    ssh -Y jbhagat@localhost -p 9999
    conda activate aeon_env
    jupyter-lab --no-browser --ip 192.168.234.1 --port 2222
    ```

3) In a new terminal, forward the port the jupyter hub server is running on to a new port on the localhost: 
    ```
    ssh -Y jbhagat@localhost -p 9999 -L localhost:2222:192.168.234.1:2222
    ```

4) In the browser, open up the jupyter hub server via the localhost address:
    `localhost:2222`

See `docs/examples/dj_example_notebook.ipynb` for an example notebook that you can play with in the jupyterhub session.
