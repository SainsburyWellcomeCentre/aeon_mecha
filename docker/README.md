# Aeon container environment

## Server-side setup

1. SSH into the SWC server then SSH again to the `aeon-db2` vm. For example, if your server credentials are stored at `~/.ssh/id` and if your username is `jburling` ...

`ssh -i ~/.ssh/id jburling@aeon-db2 -J jburling@ssh.swc.ucl.ac.uk`

2. Add the ssh deploy key to `~/.ssh/aeon_mecha` once connected to the server so that you can clone the private repo (see steps below).
   - Copy private deploy key from lastpass
   - Set `export GITHUB_DEPLOY_KEY=` in the terminal (with your pasted key after `=`).
   - Run the command below to create all the necessary ssh files

```bash
mkdir ~/.ssh/ && \
chmod 700 ~/.ssh && \
touch ~/.ssh/config && \
chmod 600 ~/.ssh/config && \
touch ~/.ssh/known_hosts && \
chmod 600 ~/.ssh/known_hosts && \
echo "${GITHUB_DEPLOY_KEY}" > ~/.ssh/aeon_mecha && \
chmod 600 ~/.ssh/aeon_mecha && \
echo -e "Host * \n  AddKeysToAgent yes\n  IdentityFile ~/.ssh/aeon_mecha\n" >> \
  ~/.ssh/config && \
ssh-keyscan github.com >> ~/.ssh/known_hosts
```

3. Clone the repo to some directory on the server (substitute `vathes` for whatever fork you're working with).

   - `git clone git@github.com:vathes/aeon_mecha.git --branch datajoint_pipeline`
   - _Note_: If you use your own fork instead of the one above, just know that the container image will be fixed to whatever is built within the GitHub actions file `.github/workflows/docker-aeon-ingest.yml`.

## Docker usage

Download an image with `aeon_mecha` installed and start the database operations by using the `ingest_high` and `ingest_mid` container services in the docker compose file.

### `docker/docker-compose.yml`

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
DJ_USER=jburling
DJ_PASS=*******
```

3. Edit the `command: ...` fields for the services `ingest_high`, `ingest_mid` that run high and mid ingestion priorities. You may want to change the appropriate `sleep` and `duration` settings for each priority. Comment the `command: ` lines for each service if you want to run the container indefinitely and doing nothing, this will use the default command found in `x-aeon-ingest-common`.

4. Log in to authenticate using your PAT (personal access token). Export your token to the variable `CR_PAT`.

```bash
export CR_PAT=YOUR_TOKEN
echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
```

5. Run docker compose
   - `docker-compose up -d`
   - `sudo docker-compose up -d` if `sudo` privileges are required.

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

- Github user that stores the docker image to pull.

### `docker/image/`

This section describes how to manually build the container image locally. The step isn't necessary because the `docker-compose.yml` file will pull the prebuilt image from GitHub.

The file `Dockerfile` will create an image that copies the private repo content and sets up conda and necessary packages, as well as install the `aeon_mecha` python package.

The Dockerfile makes use of `buildkit`. To push multiple architectures at a time, you need `buildx`. You may need to set this up before trying to build the image.

```bash
docker buildx install
docker buildx create --platform linux/arm64,linux/amd64 --name=mrbuilder --use
```

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
