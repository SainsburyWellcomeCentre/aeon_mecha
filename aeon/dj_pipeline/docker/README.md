# Dockerfile

The file `Dockerfile` will create an image that clones a private repo in the first stage and sets up conda and necessary packages in the second stage as well as install the python package from the private repo `aeon_mecha`. 

A deploy key is used and must be set up on github.com to clone the repo, if not done so already. The key contents are passed as a build argument to the first stage and not accessible from the target image. 

## Setup

1. SSH into the SWC server then again to the `aeon-db2` vm. For example, if your server credentials are stored at `~/.ssh/id` and if your username is `jburling` ...
    - `ssh -i ~/.ssh/id jburling@aeon-db2 -J jburling@ssh.swc.ucl.ac.uk`

2. Add the ssh deploy key to `~/.ssh/aeon_mecha` once connected to the server so that you can clone the private repo (see steps below).
    - Copy private key from lastpass
    - Set `export SSH_KEY=` in the terminal with your pasted key after `=`
    - Run the command below to create all the necessary ssh files

```bash
mkdir ~/.ssh/ && \
chmod 700 ~/.ssh && \
touch ~/.ssh/config && \
chmod 600 ~/.ssh/config && \
touch ~/.ssh/known_hosts && \
chmod 600 ~/.ssh/known_hosts && \
echo "${SSH_KEY}" > ~/.ssh/aeon_mecha && \
chmod 600 ~/.ssh/aeon_mecha && \
echo "Host * \n  AddKeysToAgent yes\n  IdentityFile ~/.ssh/aeon_mecha\n" >> ~/.ssh/config && \
ssh-keyscan github.com >> ~/.ssh/known_hosts
```

3. Clone the repo to some directory on the server. 
    - `git clone git@github.com:vathes/aeon_mecha.git --branch datajoint_pipeline --single-branch`
    - *Note*: If you use your own fork instead of the one above, just know that the container will be fixed to whatever is set in the `.env` file or `docker-compose.yml` file with the `GITHUB_USER` variable.

4. Navigate to `aeon_mecha/aeon/dj_pipeline/docker` after cloning repo.

5. Create a `.env` file to be used with `docker-compose.yml`.

```bash
touch .env
cat template.env >> .env
```

6. Edit the `.env` environment variables

```bash
LOCAL_CEPH_ROOT=/ceph/aeon
DJ_USER=jburling
DJ_PASS=*******
DJ_HOST=host.docker.internal
```

7. Append `SSH_KEY` to `env` file (because `sudo`).
    - `echo "SSH_KEY=$(awk -v ORS='\\n' '1' ~/.ssh/aeon_mecha)" >> .env`
  
<!-- echo "IMAGE_CREATED=$(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> .env -->

8. Run docker compose
    - `sudo docker-compose up -d`

<!-- 

### Editing `docker-compose.yml`

1. In the sections `x-ceph-volume` and `x-djstore-volume`, change the paths specified in `source: ` to point to where container paths should be mounted on the host machine (also see "Mounting volume on remote server using SSHFS" below). The paths on your local machine must exist first.

2. Edit the contents of `DJ_USER` and `DJ_PASS` in the section `environment:` so that DataJoint can connect to the `aeon-eb` database using your username and password (also see section "Using DataJoint in the docker container to access the database" below).

3. Uncomment and edit the `command: ` fields for the services `aeon_high`, `aeon_mid`, and `aeon_low`. Change to the appropriate `sleep`, `duration`, and `max_calls` settings given the priority. The default command is to show the help documentation for the `aeon_ingest` script then exit. Change the command to `tail -f /dev/null` to have the container run continuously so that you can enter it.

4. (optional) To use your own fork of the repo, change `GITHUB_USER` to your username. Make sure Docker can access the repo by setting up the deploy key in the repo settings. 





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

-->
