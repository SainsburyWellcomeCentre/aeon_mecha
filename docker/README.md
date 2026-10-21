# Aeon container environment

Use of the container requires 1) access to the raw data and optionally, 2) access to the database. See server-side setup below.

## Server-side setup

1. SSH into the SWC server then SSH again to the `aeon-db2` vm. For example, if your server credentials are stored at `~/.ssh/id` and if your username is `jburling` ...

`ssh -i ~/.ssh/id jburling@aeon-db2 -J jburling@ssh.swc.ucl.ac.uk`

2. Add the ssh deploy key to `~/.ssh/aeon_mecha` once connected to the server so that you can clone the private repo (see steps below).
   - Copy private deploy key from lastpass
   - Set `GITHUB_DEPLOY_KEY=` in the terminal (with your pasted key after `=`). You can also skip setting this variable and the last line in the code below and copy and paste the key directly.
   - Run the commands below to create all the necessary ssh files

```bash
mkdir ~/.ssh/
chmod 700 ~/.ssh
touch ~/.ssh/config
chmod 600 ~/.ssh/config
touch ~/.ssh/known_hosts
chmod 600 ~/.ssh/known_hosts
echo -e "Host * \n  AddKeysToAgent yes\n  IdentityFile ~/.ssh/aeon_mecha\n" >> ~/.ssh/config
ssh-keyscan github.com >> ~/.ssh/known_hosts
echo "${GITHUB_DEPLOY_KEY}" > ~/.ssh/aeon_mecha
chmod 600 ~/.ssh/aeon_mecha
```

Optional, store personal public key.

```bash
touch ~/.ssh/id.pub
chmod 644 ~/.ssh/id.pub
echo "my-key-string" >> ~/.ssh/id.pub
```

3. Clone the repo to some directory on the server (substitute `SainsburyWellcomeCentre` for whatever fork you're working with).

   - `git clone https://github.com/SainsburyWellcomeCentre/aeon_mecha.git --branch datajoint_pipeline`
   - _Note_: If you use your own fork instead of the one above, just know that the container image will be a snapshot of whatever is defined within the GitHub actions file [`.github/workflows/docker-aeon-mecha.yml`](../.github/workflows/docker-aeon-mecha.yml).

## Local SSH setup

### SSH into `aeon_db` dj worker

To SSH directly as the `aeon_db` user on the `aeon-db2` vm, setup your `~/.ssh/config` on your local machine, adding the following lines (this only works if your user has sudo privileges to switch to `aeon_db`):

```bash
Host swc
  HostName ssh.swc.ucl.ac.uk
  User myusername
  IdentityFile ~/.ssh/id

Host aeon_djworker
  HostName aeon-db2
  User myusername
  ProxyCommand ssh -q -W %h:%p swc
  RemoteCommand sudo -Sv < pwd.txt; sudo -iu aeon_db
  RequestTTY yes
```

The file `~/.ssh/id` is your key to connect to the server. The file `pwd.txt` is a file with your user password, placed in your user's home directory on `aeon-db2`, e.g., `myusername@aeon-db2`. You can do `chmod 400 ~/pwd.txt` after it is saved to the file. Alternatively, you can substitute the `RemoteCommand` line to include your password (less safe) instead of using a file, like so: `sudo -Sv <<< "mypassword"; sudo -iu aeon_db`.

Connect by running `ssh aeon_djworker` in a terminal.

To run a remote `vscode` session using the [`Remote Development`](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) extension, run the following line:

```bash
code --folder-uri vscode-remote://ssh-remote+aeon_djworker/nfs/nhome/live/aeon_db
```

You can also save that as an alias or a shell script.

### Forwarding the database port to your local machine

Add the following lines to your ssh config file (which also has the `Host swc` section from above).

```bash
Host aeon-database
  HostName hpc-gw1
  User myusername
  LocalForward 127.0.0.1:3307 aeon-db2:3306
  ProxyJump swc
  ControlMaster auto
  ControlPath ~/.ssh/controlmasters/%r@%h:%p
```

Connect by running `ssh aeon-database` in a terminal.

Check and stop the connection by running `ssh -O check aeon-database` and `ssh -O stop aeon-database` in a terminal, respectively.

The database will be accessible at `localhost:3307` or `127.0.0.1:3307` (change 3307 in the config to use a different port).

## Docker usage

### `docker/docker-compose.yml`

The following will download an image containing `aeon_mecha` pre-installed and start the database operations by using the `ingest_high` and `ingest_mid` container services in the docker compose file.

Since the container is on a private repo, you'll need to be able to use the command `docker login` and to also create a personal access token to pull the image from `ghcr.io`, see here: [creating a PAT](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) and [working with the container registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

Navigate to `aeon_mecha/docker` after cloning the repo contents.

The file `docker-compose.yml` requires that you create a `.env` file to setup the data paths and parts of the DataJoint configuration.

1. Create the `.env` file that will be used with `docker-compose.yml`.

```bash
touch .env
cat template.env >> .env
```

2. Edit the `.env` environment variables (see description section below for more info), for example:

```bash
LOCAL_CEPH_ROOT=/ceph/aeon
LOCAL_DJ_STORE=/ceph/aeon/aeon/dj_store
DJ_HOST=host.docker.internal
DJ_USER=my_swc_username
DJ_PASS=*******
```

3. Edit the `command: ...` fields for the services `acquisition_worker`, `streams_worker` that run different parts of the pipeline. You may want to change the appropriate `sleep` and `duration` settings for each priority. Comment out the `command: ` lines for each service if you want to run the container indefinitely and doing nothing, this will use the default command found in `x-aeon-ingest-common`.

4. Log in to authenticate using your PAT (personal access token). Export your token to the variable `CR_PAT`.

```bash
export CR_PAT=YOUR_TOKEN
echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
```

5. Run docker compose from the `aeon_mecha/docker/` subdirectory

```bash
docker-compose up -d
```

#### Description of `.env` variables

`LOCAL_CEPH_ROOT`

- Path to the raw data directory on the host machine.

`LOCAL_DJ_STORE`

- Path to the exported data from DataJoint tables.

`DJ_USER`

- DataJoint username used to connect to the database.

`DJ_PASS`

- DataJoint password used to connect to the database.

`DJ_HOST`

- Database hostname/url.

`GITHUB_REPO_OWNER`

- Github user that stores the docker image/package from which to pull. User must have a PAT for ghcr login.

### `docker/image/`

This section describes how to manually build the container image locally. The step isn't necessary because the `docker-compose.yml` file will pull the pre-built image from GitHub packages.

The file `Dockerfile` will create an image that copies the private repo content and sets up conda and necessary packages, as well as install the `aeon_mecha` python package. The Dockerfile also makes use of `buildkit`. To push multiple architectures at a time, you need `buildx`. You may need to set this up before trying to build the image.

```bash
docker buildx install
docker buildx create --platform linux/arm64,linux/amd64 --name=mrbuilder --use
```

Change `arm64` to `amd64` if required, or another supported build platform.

```bash
cd aeon_mecha
VERSION=v0.0.0a
DATE_CREATED=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
PLATFORM=arm64
docker buildx build \
    --file=./docker/image/Dockerfile \
    --output=type=docker \
    --platform=linux/$PLATFORM \
    --tag=aeon_mecha:$VERSION \
    --tag=aeon_mecha:latest \
    --build-arg IMAGE_CREATED=$DATE_CREATED \
    --build-arg IMAGE_VERSION=$VERSION \
    .
```

To remove the installed buildx builder

```bash
docker buildx rm mrbuilder
docker buildx uninstall
```
