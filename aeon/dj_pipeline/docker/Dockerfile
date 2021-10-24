# start cloning repo using private key ---------------------------------------
# - multi-stage build is used so as not to store private key in history when entered as a build arg
# - this stage should be edited and merged with last stage when repo becomes public

FROM debian:latest as private_repo_clone

LABEL maintainer="Joseph Burling"

ARG SSH_KEY

# github repo info for cloning
ARG GITHUB_USER
ARG GITHUB_REPO
ARG GITHUB_BRANCH

# need git for clone and ssh-keyscan
RUN apt-get -yq update && \
    apt-get -yqq install openssh-client git

# - make ssh directory under root
# - set required permissions for ssh dir
# - make ssh config file
# - set required permissions for ssh config
# - make known_hosts file
# - set required permissions for known_hosts
# - write private key to local github key file
# - set required private key permissions
# - automatically load keys into the ssh-agent
# - add GitHub to known hosts to bypass perm add key prompt
RUN mkdir /root/.ssh/ && \
    chmod 700 /root/.ssh && \
    touch /root/.ssh/config && \
    chmod 600 /root/.ssh/config && \
    touch /root/.ssh/known_hosts && \
    chmod 600 /root/.ssh/known_hosts && \
    echo "${SSH_KEY}" > /root/.ssh/github && \
    chmod 600 /root/.ssh/github && \
    echo "Host * \n  AddKeysToAgent yes\n  IdentityFile /root/.ssh/github\n" \
    >> /root/.ssh/config && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

# start cloning private github repo for copying at later stage
WORKDIR /tmp
RUN git clone git@github.com:${GITHUB_USER}/${GITHUB_REPO}.git --branch ${GITHUB_BRANCH} --single-branch


# start main conda build stage -----------------------------------------------
# - setup and use custom conda environment

FROM datajoint/djbase:py3.9-debian

ARG GITHUB_REPO

ENV RAW_DATA_DIR=/ceph/aeon

# change to working folder in home directory for pip install
WORKDIR /home/anaconda

# chown git repo to user
COPY --chown=anaconda:anaconda --from=private_repo_clone \
    /tmp/${GITHUB_REPO} \
    ${GITHUB_REPO}

# copy over global datajoint config file w/ hard-coded paths
COPY --chown=anaconda:anaconda .datajoint_config.json .

# create the conda environment `aeon_env` from yml file
RUN conda env create --force --file ${GITHUB_REPO}/env.yml

# make following RUN commands use the `aeon_env` env:
SHELL ["conda", "run", "-n", "aeon_env", "/bin/bash", "-c"]

# install python package as dev pkg
RUN pip install ${GITHUB_REPO}/.

# template command to change in docker-compose.yml
CMD ["conda", "run", "-n", "aeon_env", "aeon_ingest", "--help"]
