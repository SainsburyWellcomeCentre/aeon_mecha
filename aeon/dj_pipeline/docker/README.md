<!--
# Random commands

docker-compose build --build-arg SSH_KEY="$(cat ~/.ssh/aeon_mecha)" && docker-compose up -d

docker run --name test_run -it aeon_ingest

docker exec -it containerized_ingest_aeon_high_1 bash

docker build --target private_repo_clone -t test_repo --build-arg SSH_KEY="$(cat ~/.ssh/aeon_mecha)" .

docker builder prune --all -f
-->


## Dockerfile

The file `Dockerfile` will create an image that clones a private repo in the first stage and sets up conda and necessary packages in the second stage as well as install the python package from the private repo `aeon_mecha`. 

A deploy key is used and must be set up on github.com to clone the repo, if not done so already. The key contents are passed as a build argument to the first stage and not accessible from the target image. 

### Editing `docker-compose.yml`

1. In the sections `x-ceph-volume` and `x-djstore-volume`, change the paths specified in `source: ` to point to where container paths should be mounted on the host machine (also see "Mounting volume on remote server using SSHFS" below). The paths on your local machine must exist first.

2. Edit the contents of `DJ_USER` and `DJ_PASS` in the section `environment:` so that DataJoint can connect to the `aeon-eb` database using your username and password (also see section "Using DataJoint in the docker container to access the database" below).

3. Uncomment and edit the `command: ` fields for the services `aeon_high`, `aeon_mid`, and `aeon_low`. Change to the appropriate `sleep`, `duration`, and `max_calls` settings given the priority. The default command is to show the help documentation for the `aeon_ingest` script then exit. Change the command to `tail -f /dev/null` to have the container run continuously so that you can enter it.

4. (optional) To use your own fork of the repo, change `GITHUB_USER` to your username. Make sure Docker can access the repo by setting up the deploy key in the repo settings. 


### Build the image and run the `aeon_ingest` python script in multiple containers

To make `docker-compose` easier to use, you can use the shell script `compose.sh` and pass it your deploy key to use when cloning the repo. The `compose.sh` script will use the commands specified in `docker-compose.yml` when running the python script `aeon_ingest`.

Running `./compose.sh -h` will show the following:

```
compose.sh : start/stop containerized aeon ingestion routine

Usage: compose.sh [options]

options:
    -h, --help              show usage help
    -d, --down              docker compose down, removing orphans
    -k, --key=DEPLOY_KEY    specify path to private deploy key (default=~/.ssh/aeon_mecha)
        --low=N_WORKERS_L   number of workers for low priority tasks
        --mid=N_WORKERS_M   number of workers for mid priority tasks
        --high=N_WORKERS_H  number of workers for high priority tasks
```

**`compose.sh` usage examples**:

```bash
# build only, don't run
./compose.sh --key=~/.ssh/aeon_mecha 

# build and run two aeon_high services simultaneously 
./compose.sh --key=~/.ssh/aeon_mecha --high=2

# build and run two aeon_high services and one aeon_low service simultaneously 
./compose.sh --key=~/.ssh/aeon_mecha --high=2 --low=1

# take down all running services and remove container
./compose.sh --down

# take down and rebuild but don't run
./compose.sh --down --key=~/.ssh/aeon_mecha 

# take down, rebuild, and run workers
./compose.sh --down --key=~/.ssh/aeon_mecha --high=2 ...
```



## Test locally

### Setup SSH config

In your ssh config file at `~/.ssh/config`, setup access to _hpc-gw1_ (via jump) and _aeon-db_ database. This assumes the keys `local_key` and `swc_key` have already been added using `ssh-keygen` and copied over using `ssh-copy-id`. Add the lines below (replace `<user>` with your username):

```bash
# > ssh -i ~/.ssh/swc_key <user>@hpc-gw1.hpc.swc.ucl.ac.uk
Host aeon
  HostName hpc-gw1.hpc.swc.ucl.ac.uk
  User <user>
  ProxyCommand ssh -q -W %h:%p swc

# > ssh -v -N -f -M -S ~/.ssh/controlmasters/%r@%h:%p <user>@hpc-gw1.hpc.swc.ucl.ac.uk -J <user>@ssh.swc.ucl.ac.uk -L 127.0.0.1:3306:aeon-db:3306
Host aeon-db
  HostName hpc-gw1.hpc.swc.ucl.ac.uk
  User <user>
  LocalForward 127.0.0.1:3306 aeon-db:3306
  ProxyJump swc
  ControlMaster auto
  ControlPath ~/.ssh/controlmasters/%r@%h:%p

# > ssh -i ~/.ssh/local_key <user>@ssh.swc.ucl.ac.uk
Host swc
  HostName ssh.swc.ucl.ac.uk
  User <user>
  IdentityFile ~/.ssh/local_key
```

### Using DataJoint in the docker container to access the database

1. In the file `.env` (same location as `docker-compose.yml`), change the environment variables `DJ_USER` and `DJ_PASS` to your _aeon-db_ username and password. The user name should be the same as your login name to _hpc-gw1_.

2. Using the `aeon-db` hostname in your ssh config file, connect to the Aeon database (_aeon-db_) and forward port 3306 on _hpc-gw1_ to 3306 on your local machine. You will be asked to enter your password if keys are not found in `authorized_keys` on the remote server. 

```bash
ssh -v -N -f aeon-db
```

To stop or check on the connection:

```bash
ssh -O check aeon-db
ssh -O stop aeon-db
```

### Mounting volume on remote server using SSHFS

1. Install SSHFS (see [here](https://code.visualstudio.com/docs/remote/troubleshooting#_using-sshfs-to-access-files-on-your-remote-host) for one example).

2. Allow `/ceph/aeon` on the remote server to be accessed on local machine using `sshfs`.

```bash
# Make the local directory where the remote filesystem will be mounted
mkdir -p "$HOME/SSHFS/aeon/ceph/aeon"

# Mount the remote filesystem
sshfs "aeon-db:/ceph/aeon" "$HOME/SSHFS/aeon/ceph/aeon" -ovolname=aeon -o workaround=nonodelay -o transform_symlinks -o idmap=user -C
```

Make sure the local path set above matches the path used to map to `/ceph/aeon` in `docker-compose.yml`.

3. The path in the container `/home/anaconda/djstore` set in `docker-compose.yml` should mount to somewhere on local drive to test saving external storage entries.
