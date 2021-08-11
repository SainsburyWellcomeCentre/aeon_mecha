The examples below use the SWC HPC username `jbhagat`. When you run the commands, replace this username with your username.

## Connecting to the dashboard:

### On SWC's intranet

In your browser, navigate to `192.168.240.50:8050`

### Outside of SWC's intranet

1) In a terminal, set up SSH local port forwarding to HPC-GW1 on a port that is not currently in use:
    `ssh -L 9998:hpc-gw1:22 jbhagat@ssh.swc.ucl.ac.uk`

2) In the browser, set up a proxy server with the SOCKS5 protocol to go to localhost (server 127.0.0.1) via the forwarded port (9998). (On Chrome this can be done via the "SwitchyOmega" chrome extension)

![SwitchOmega screengrab](switchyomega_chrome_extension.png)

3) In the browser, navigate to `192.168.240.50:8050`