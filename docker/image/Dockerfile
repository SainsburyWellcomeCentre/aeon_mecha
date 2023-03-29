# user specified build arguments
ARG NEW_USER_NAME=aeon_db
ARG NEW_USER_GROUP=aeon_db
ARG NEW_USER_UID=6001
ARG NEW_USER_GID=6001
ARG NEW_USER_SUDO=true
ARG AEON_GROUP_NAME=aeon
ARG AEON_GROUP_GID=801100510
ARG IMAGE_CREATED=2021-11-11T11:11:11Z
ARG IMAGE_VERSION=v0.0.0a

# main build stage =====================================================================
FROM ghcr.io/ttngu207/conda_base:latest as aeon_mecha_docker_pre

# inherit global args
ARG NEW_USER_NAME
ARG NEW_USER_GROUP
ARG NEW_USER_UID
ARG NEW_USER_GID
ARG NEW_USER_SUDO
ARG AEON_GROUP_NAME
ARG AEON_GROUP_GID
ARG IMAGE_CREATED
ARG IMAGE_VERSION

# default paths
ARG CEPH_ROOT=/ceph/aeon
ARG DJ_EXT_STORE=/ceph/aeon/aeon/dj_store
ARG AEON_PKG=/aeon_mecha
ARG NEW_USER_HOME=/home/${NEW_USER_NAME}

# bash as default shell
SHELL ["/bin/bash", "-lc"]

# create new user from the NEW_USER_* arguments
RUN init-new-user

# hack for 32-bit id's
RUN rm /var/log/lastlog /var/log/faillog && \
    ln -s /dev/null /var/log/lastlog && \
    ln -s /dev/null /var/log/faillog

# add aeon group, append group to new user
RUN groupadd --gid ${AEON_GROUP_GID} --force ${AEON_GROUP_NAME} && \
    usermod -a -G ${AEON_GROUP_NAME} ${NEW_USER_NAME}

# copy over repo content and chown to new user
COPY --chown=${NEW_USER_NAME}:${AEON_GROUP_NAME} ./ ${AEON_PKG}/
USER root:${AEON_GROUP_NAME}

# create the aeon and dj_store data directories
RUN mkdir -p ${CEPH_ROOT} ${DJ_EXT_STORE} && \
    chown -R ${NEW_USER_NAME}:${AEON_GROUP_NAME} ${CEPH_ROOT} ${DJ_EXT_STORE} && \
    chmod -R 6775 ${CEPH_ROOT} ${DJ_EXT_STORE} ${AEON_PKG}

# copy apt-get dependencies, conda environment yml, datajoint config template
RUN cp -f ${AEON_PKG}/docker/image/apt_requirements.txt /srv/conda/apt_requirements.txt && \
    cp -f ${AEON_PKG}/env.yml /srv/conda/environment.yml && \
    cp -f ${AEON_PKG}/docker/image/.datajoint_config.json /tmp/.datajoint_config.json

# create the local and global datajoint config files, hard-coding the data paths
RUN awk -v DJSTORE="$DJ_EXT_STORE" -v CEPHROOT="$CEPH_ROOT" \
    '{sub(/{{ DJ_EXT_STORE }}/,DJSTORE);sub(/{{ CEPH_ROOT }}/,CEPHROOT)} 1' \
    /tmp/.datajoint_config.json | tee \
    ${AEON_PKG}/dj_local_config.json \
    ${NEW_USER_HOME}/.datajoint_config.json > /dev/null && \
    chown ${NEW_USER_UID}:${NEW_USER_GID} ${NEW_USER_HOME}/.datajoint_config.json

# install package dependencies from apt_requirements.txt
RUN init-apt-deps

# create the conda environment `aeon` from yml file
RUN sudo-run \
    --switch-user=${NEW_USER_NAME} \
    --switch-group=condauser \
    --sudo-command=init-conda-env

# install aeon_mecha pkg in dev mode
RUN sudo-run \
    --switch-user=${NEW_USER_NAME} \
    --switch-group=condauser \
    pip install -e ${AEON_PKG}/.

RUN mkdir -p -m 6775 \
    ${NEW_USER_HOME}/.vscode-server/extensions \
    ${NEW_USER_HOME}/.vscode-server-insiders/extensions && \
    chown -R ${NEW_USER_NAME}:${NEW_USER_GROUP} ${NEW_USER_HOME}/.vscode-server* && \
    echo "$IMAGE_CREATED" > /tmp/$(echo $IMAGE_CREATED | awk '{gsub(/\:/,"-");print}')

# delete logs and unnecessary files copied over from this repo
RUN rm -rf \
    /var/log/lastlog /var/log/faillog \
    ${AEON_PKG}/docker /var/tmp/* /tmp/* && \
    $CONDA_ROOT/bin/conda clean -yqa && \
    touch /var/log/lastlog && \
    touch /var/log/faillog

# target build stage, squash previous layers ===========================================
FROM scratch as aeon_mecha_docker
COPY --from=aeon_mecha_docker_pre / /

ARG IMAGE_CREATED
ARG IMAGE_VERSION
ARG NEW_USER_NAME
ARG NEW_USER_GROUP

LABEL org.opencontainers.image.authors="DataJoint"
LABEL org.opencontainers.image.title="aeon_mecha_docker"
LABEL org.opencontainers.image.description="aeon_mecha pkg with conda environment aeon"
LABEL org.opencontainers.image.version="$IMAGE_VERSION"
LABEL org.opencontainers.image.created="$IMAGE_CREATED"

ENV TZ="America/Chicago"
ENV LANG=en_US.utf8
ENV LC_ALL=en_US.utf8
ENV CONDA_ENV_USER=

WORKDIR /home/${NEW_USER_NAME}

# for switching to user and having all groups when running a command
ENV USER_NAME=${NEW_USER_NAME}
ENV USER_GROUP=${NEW_USER_GROUP}
SHELL ["/bin/bash", "--login", "-c"]
ENTRYPOINT ["sudo-run"]
CMD ["tail", "-f", "/dev/null"]
