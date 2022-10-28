# Project Aeon's SciViz Deployment

## Required dependencies
If you have not done so already, please install the following dependencies:

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Running the application

#### Production deployment

To run the application in production mode, use the command at the root:
```bash
cd aeon/dj_pipeline/webapps/sciviz
SUBDOMAINS=testdev URL=datajoint.io STAGE_CERT=TRUE EMAIL=service-health@datajoint.com HOST_UID=$(id -u) docker-compose -f docker-compose-remote.yaml up -d
```
Please modify `SUBDOMAINS`, `URL`, and `STAGE_CERT` according to your own deployment configuration.

On the example above, the first two arguments are about what site you are going to host this on, the configuration shown here is for `https://testdev.datajoint.io`

The next two arguments are for web certifications. Set `STAGE_CERT=TRUE` for testing certs and set it to `FALSE` once you are confident in your deployment for production. `EMAIL` is for where notifications related to certs are going to be sent.

#### Local dev deployment

For local deployment, you need to ensure the connection to the `aeon-db2` server is established. This can be done by establishing the port forwarding as follows in a terminal:

```bash
ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no username@ssh.swc.ucl.ac.uk -L 3306:aeon-db2:3306 -N
```
(replace `username` with your own username)

Leave this terminal open and open up a new terminal to navigate to the `sciviz` folder  
```bash
cd aeon/dj_pipeline/webapps/sciviz
```
Docker compose up with the following command:
```
HOST_UID=$(id -u) docker-compose -f docker-compose-local.yaml up
```
To stop the application, use the same command as before but with `down` in place of `up -d`

## Verify the deployment

#### Production deployment

- Navigate to the server's address in a Google Chrome browser window.
- Set `Host/Database Address` to `aeon-db2`. 
- Set `Username` and `Password` to your own database user account (if you need one, please contact Project Aeon admin team).
- Click `Connect`.


#### Local dev deployment

- In a Google Chrome browser window, navigate to: [https://localhost/login](https://localhost/login)
- Set `Host/Database Address` to `host.docker.internal`. 
- Set `Username` and `Password` to your own database user account (if you need one, please contact Project Aeon admin team).
- Click `Connect`.

## Dynamic spec sheet
Sci-Viz is used to build visualization dashboards, this is done through a single spec sheet. The one for this deployment is called `specsheet.yaml`

Some notes about the spec sheet if you plan to tweak the website yourself:
- Page names under pages must have a unique name without spaces
- Page routes must be unique
- Grid names under grids must be unique without spaces
- Component names under components must be unique **but can have spaces**
- The routes of individual components must be unique
- Routes must start with a `/`
- Every query needs a restriction, below is the default one.
  - ```python
        def restriction(**kwargs):
            return dict(**kwargs)
    ```
- Overlapping components at the same (x, y) does not work, the grid system will not allow overlapping components it will wrap them horizontally if there is enough space or bump them down to the next row.
- Visit this [repo](https://github.com/datajoint/sci-viz) to learn more about Sci-Viz.