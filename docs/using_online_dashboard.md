In the examples below, replace <your_username> with your SWC HPC username.

## Connecting to the dashboard:

### On SWC's intranet

In your browser, navigate to `192.168.240.50:8050`

### Outside of SWC's intranet

1. In a shell, SSH to the HPC using the SOCKS5 protocol to listen to a local port (here 9997) that is not currently in use:
`ssh -D 9997 <your_username>@ssh.swc.ucl.ac.uk`

2. In the browser, set up a proxy server with the SOCKS5 protocol to go to localhost (server 127.0.0.1) on the forwarded port (9997). (On Chrome this can be done via the "SwitchyOmega" chrome extension)

![SwitchyOmega screengrab](switchyomega_chrome_extension.png)

3. In the browser, navigate to `192.168.240.50:8050`
