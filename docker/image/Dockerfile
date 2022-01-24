FROM ghcr.io/iamamutt/conda_base:latest

ARG IMAGE_CREATED=2021-11-11T11:11:11Z
ARG IMAGE_VERSION=v0.0.0a
LABEL org.opencontainers.image.authors="DataJoint"
LABEL org.opencontainers.image.title="aeon_mecha_aeon_env"
LABEL org.opencontainers.image.description="aeon_mecha pkg with conda environment aeon_env"
LABEL org.opencontainers.image.version="$IMAGE_VERSION"
LABEL org.opencontainers.image.created="$IMAGE_CREATED"

# user specified build arguments
ARG NEW_USER_NAME=aeon_db
ARG NEW_USER_GROUP=aeon_db
ARG NEW_USER_UID=6001
ARG NEW_USER_GID=6001
ARG NEW_USER_SUDO=true
ARG AEON_GROUP_NAME=aeon
ARG AEON_GROUP_GID=801100510

# default paths
ARG CEPH_ROOT=/ceph/aeon
ARG DJ_EXT_STORE=/ceph/aeon/aeon/dj_store
ARG AEON_PKG=/home/${NEW_USER_NAME}/aeon_mecha

# for the sudoing to user and having all groups entrypoint
ENV USER_NAME=${NEW_USER_NAME}
ENV USER_GROUP=${NEW_USER_GROUP}

# create new user and add aeon group, append groups to user
RUN init-new-user && \
    rm /var/log/lastlog /var/log/faillog && \
    ln -s /dev/null /var/log/lastlog && \
    ln -s /dev/null /var/log/faillog && \
    groupadd --gid ${AEON_GROUP_GID} --force ${AEON_GROUP_NAME} && \
    usermod -a -G ${AEON_GROUP_NAME} ${NEW_USER_NAME} && \
    rm /var/log/lastlog /var/log/faillog && \
    touch /var/log/lastlog && \
    touch /var/log/faillog

# copy over repo content and chown to new user
COPY --chown=${NEW_USER_NAME}:${NEW_USER_GROUP} ./ ${AEON_PKG}/
USER root:${AEON_GROUP_NAME}
WORKDIR /home/${NEW_USER_NAME}

# 1. create the aeon and dj_store data directories
# 2. copy over:
#      - entrypoint script
#      - apt-get dependencies and conda environment files
#      - global user datajoint config file hard-coding the data paths
# 3. delete unnecessary files
# 4. create the conda environment `aeon_env` from yml file and install aeon_mecha pkg
RUN mkdir -p ${CEPH_ROOT} ${DJ_EXT_STORE} && \
    chown -R :${AEON_GROUP_NAME} ${CEPH_ROOT} ${DJ_EXT_STORE} && \
    chmod -R 6774 ${CEPH_ROOT} ${DJ_EXT_STORE} ${AEON_PKG} && \
    cp -f "${AEON_PKG}/docker/image/sudo-conda-run" /usr/local/bin/sudo-conda-run && \
    chmod 755 /usr/local/bin/sudo-conda-run && \
    cp -f "${AEON_PKG}/docker/image/apt_requirements.txt" /srv/conda/apt_requirements.txt && \
    cp -f "${AEON_PKG}/env.yml" /srv/conda/environment.yml && \
    cp -f "${AEON_PKG}/docker/image/.datajoint_config.json" /tmp/.datajoint_config.json && \
    rm -rf "${AEON_PKG}/docker" && \
    awk -v DJSTORE="$DJ_EXT_STORE" -v CEPHROOT="$CEPH_ROOT" \
    '{sub(/{{ DJ_EXT_STORE }}/,DJSTORE);sub(/{{ CEPH_ROOT }}/,CEPHROOT)} 1' \
    /tmp/.datajoint_config.json > /home/${NEW_USER_NAME}/.datajoint_config.json && \
    chown ${NEW_USER_NAME}:${NEW_USER_GROUP} /home/${NEW_USER_NAME}/.datajoint_config.json && \
    init-env pip install ${AEON_PKG}/.

ENTRYPOINT ["sudo-conda-run"]
CMD ["tail", "-f", "/dev/null"]
