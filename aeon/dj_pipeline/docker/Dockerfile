ARG GITHUB_REPO
ARG IMAGE_CREATED=2021-11-11T11:11:11Z

# start cloning repo using private key -------------------------------------------------
# - multi-stage build is used to not store private key in history
#   when entered as a build arg
# - TODO this stage should be edited and merged with last stage when repo becomes public
FROM debian:stable-slim as private_repo_clone

ARG IMAGE_CREATED
LABEL org.opencontainers.image.created="$IMAGE_CREATED"

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
RUN git clone git@github.com:${GITHUB_USER}/${GITHUB_REPO}.git \
    --branch ${GITHUB_BRANCH} --single-branch

# start main conda build stage ---------------------------------------------------------
# - setup and use custom conda environment
FROM iamamutt/conda_base:latest

ARG IMAGE_CREATED
ARG IMAGE_VERSION=v0.0.0a
ARG GITHUB_REPO
ARG USER_NAME=condauser
LABEL org.opencontainers.image.authors="DataJoint"
LABEL org.opencontainers.image.title="aeon_mecha_datajoint_pipeline"
LABEL org.opencontainers.image.description="Run aeon-db ingestion routines."
LABEL org.opencontainers.image.version="$IMAGE_VERSION"
LABEL org.opencontainers.image.created="$IMAGE_CREATED"

WORKDIR /home/${USER_NAME}

# copy over git repo and chown to user
COPY --chown=${USER_NAME} --from=private_repo_clone \
    /tmp/${GITHUB_REPO} \
    ${GITHUB_REPO}

# copy over global datajoint config file w/ hard-coded paths
COPY --chown=${USER_NAME} .datajoint_config.json .

# create the mounted directories
ENV RAW_DATA_DIR=/ceph/aeon DJ_EXTERNAL_STORE=/home/${USER_NAME}/djstore
RUN mkdir -p ${RAW_DATA_DIR} ${DJ_EXTERNAL_STORE} && \
    chown -R ${USER_NAME} ${RAW_DATA_DIR} ${DJ_EXTERNAL_STORE}

# directory where default apt_requirements.txt and environment.yml are located
COPY ["nodeps.txt", "apt_requirements.txt*", "environment.yml*", "/srv/conda/"]

# create the conda environment `aeon_env` from yml file and install aeon_mecha pkg
RUN CONDA_ENV_FILE=${GITHUB_REPO}/env.yml init-env pip install ${GITHUB_REPO}/.

USER ${USER_NAME}
SHELL ["/bin/bash", "--login", "-c"]
ENTRYPOINT ["conda-run"]
CMD ["tail", "-f", "/dev/null"]
