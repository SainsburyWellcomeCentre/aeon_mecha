# start cloning repo using private key ---------------------------------------
# - multi-stage build is used so as not to store private key in history when entered as build arg
# - this stage should be edited and merged with last stage when repo is public

FROM debian:latest as private_repo_clone

LABEL maintainer="Joseph Burling"

# set SSH_KEY during build with `--build-arg SSH_KEY="$(cat ~/.ssh/github)"`
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
# - setup and use custom conda environment as default

FROM debian:latest

LABEL maintainer="Joseph Burling"

# see docker-compose for project specific args
ARG USER_NAME=datajoint
ARG USER_GROUP=datajoint
ARG PROJECT_NAME
ARG RAW_DATA_PATH
ARG GITHUB_REPO

# conda stuff
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV CONDA_DIR=/home/${USER_NAME}/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# volume mapping vars
ENV USER_DATA_DIR=/home/${USER_NAME}/${PROJECT_NAME}
ENV RAW_DATA_DIR=${RAW_DATA_PATH}

# set the following in .env file
ENV DJ_USER="root"
ENV DJ_PASS=""
ENV DJ_HOST="host.docker.internal"


# install minimal deps
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        git \
        openssh-client \
        wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# create new user under custom group
RUN groupadd ${USER_GROUP} && \
    useradd -ms /bin/bash -g ${USER_GROUP} ${USER_NAME} && \
    chown -R ${USER_NAME}:${USER_GROUP} /home/${USER_NAME}
USER ${USER_NAME}:${USER_GROUP}
WORKDIR /home/${USER_NAME}

# download latest miniconda version
RUN wget --no-hsts --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-$(uname -m).sh -O /tmp/miniconda.sh

# begin user miniconda install
RUN /bin/bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
	rm -rf /tmp/miniconda.sh

# start root actions
USER 0

# raw data dir is expected to be mounted to read-only dir on host machine
RUN mkdir -p ${RAW_DATA_DIR} && \
    chown -R ${USER_NAME}:${USER_GROUP} ${RAW_DATA_DIR} && \
    chmod -R 0444 ${RAW_DATA_DIR}

# copy repo from prev stage to ${USER_DATA_DIR}/${GITHUB_REPO}
COPY --from=private_repo_clone \
     /tmp/${GITHUB_REPO} \
     ${USER_DATA_DIR}/${GITHUB_REPO}

# remove git stuff, chown to user
RUN rm -rf ${USER_DATA_DIR}/${GITHUB_REPO}/.git* && \
    chown -R ${USER_NAME}:${USER_GROUP} ${USER_DATA_DIR}/${GITHUB_REPO}

# start user actions
USER ${USER_NAME}:${USER_GROUP}

# - update base conda pkg
# - create the conda environment `aeon_env` from yml file
# - cleanup all, rm unnecessary static libs and bytecode caches
# - activate `aeon_env` for bash shell
RUN conda update -y -n base -c defaults conda && \
    conda env create --force -f ${USER_DATA_DIR}/${GITHUB_REPO}/env.yml && \
    conda clean -afy && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate aeon_env" \
      >> ~/.bashrc

# change to working folder in home directory for pip install
WORKDIR ${USER_DATA_DIR}

# make following RUN commands use the `aeon_env` env:
SHELL ["conda", "run", "-n", "aeon_env", "/bin/bash", "-c"]

# check if the env is activated and required packages installed
RUN python -c "import datajoint"

# install package as dev pkg
RUN pip install -e ${GITHUB_REPO}

# check if pkg installed
RUN python -c "from aeon import dj_pipeline"

# container executes cmd using conda environment
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "aeon_env", "python"]

CMD ["tail", "-f", "/dev/null"]
