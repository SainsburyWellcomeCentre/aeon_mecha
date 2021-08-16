<!--
# Random commands

docker-compose build --build-arg SSH_KEY="$(cat ~/.ssh/github)" && docker-compose up -d

docker run --name test_run -it aeon_ingest

docker build --target private_repo_clone -t test_repo --build-arg SSH_KEY="$(cat ~/.ssh/github)" .
-->

TODO:

- Set ENTRYPOINT to the ingestion routine script and CMD to script parameters.
- Update Vathes fork so as to allow aeon pkg install
- Add deploy key to vathes fork
- Add process.py as a package script
- Edit allowable input args to process.py script

## Build Docker image

The file `Dockerfile` will create an image that clones a private repo in the first stage and installs conda and necessary packages in the second stage and the python package from the private repo. 

A deploy key is used and must be set up on github.com to clone the repo. The key contents are passed as a build argument to the first stage and not accessible from the target image. 

1. Edit contents of `docker-compose.yml`
 
    - To use your own fork of the repo, change `GITHUB_USER` to your username.
 
    - In the section `volumes`, change the paths left of the `:` to point to where container paths should be mounted (see "Mounting volume on remote server using SSHFS" below).  

2. Edit the contents of the `.env` file so that DataJoint can access database info.

    - See section "Using DataJoint in the docker container to access the database" below.

3. Build and run the image

To make `docker-compose` easier to use, you can use the shell script `compose.sh` and pass it your deploy key to use when cloning the repo. Running `./compose.sh -h` will show the following:

```
NOTE: Change to location of 'docker-compose.yml' before running

usage: ./compose.sh "~/.ssh/my_deploy_key"
  if ./compose.sh has no input argument, the ssh key is pulled from ~/.ssh/github
  to docker-compose down: ./compose.sh -d
```

Without `./compose.sh` you can do the following, changing `github` to whatever the name of the deploy key is.

```bash
docker-compose build --build-arg SSH_KEY="$(cat ~/.ssh/github)" && docker-compose up -d
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

Make sure the local path set above matches the path used to map `/ceph/aeon` in the file `docker-compose.yml` > `volumes`.
