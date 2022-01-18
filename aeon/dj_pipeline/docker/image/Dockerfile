FROM ghcr.io/iamamutt/conda_base:latest

ARG IMAGE_CREATED=2021-11-11T11:11:11Z
ARG IMAGE_VERSION=v0.0.0a
LABEL org.opencontainers.image.authors="DataJoint"
LABEL org.opencontainers.image.title="aeon_mecha_datajoint_pipeline"
LABEL org.opencontainers.image.description="Run aeon-db ingestion routines."
LABEL org.opencontainers.image.version="$IMAGE_VERSION"
LABEL org.opencontainers.image.created="$IMAGE_CREATED"

ARG USER_NAME=condauser
ARG PKG_SUBDIR=aeon_mecha
ARG CEPH_ROOT=/ceph/aeon
ARG BUILDER_HOME=/home/${USER_NAME}
ARG DJ_EXTERNAL_STORE=${BUILDER_HOME}/djstore
ARG PKG_CONTENT_PATH=${BUILDER_HOME}/${PKG_SUBDIR}
ARG DOCKER_IMAGE_PATH=${PKG_CONTENT_PATH}/aeon/dj_pipeline/docker/image

# create the raw and storage data directories
RUN mkdir -p ${CEPH_ROOT} ${DJ_EXTERNAL_STORE} ${PKG_CONTENT_PATH} && \
    chown -R ${USER_NAME} ${CEPH_ROOT} ${DJ_EXTERNAL_STORE} ${PKG_CONTENT_PATH}

# copy over repo content and chown to user
COPY --chown=${USER_NAME} ./ ${PKG_CONTENT_PATH}/

# create the conda environment `aeon_env` from yml file and install python pkg
RUN APT_GET_DEPS=${DOCKER_IMAGE_PATH}/apt_requirements.txt \
    CONDA_ENV_FILE=${PKG_CONTENT_PATH}/env.yml \
    init-env pip install ${PKG_CONTENT_PATH}/.

# switch to non-root user and it's home dir
USER ${USER_NAME}
WORKDIR ${BUILDER_HOME}

# copy over global user datajoint config file w/ hard-coded paths to data
RUN awk -v DJSTORE="$DJ_EXTERNAL_STORE" -v CEPHROOT="$CEPH_ROOT" \
    '{sub(/{{ DJ_EXTERNAL_STORE }}/,DJSTORE);sub(/{{ CEPH_ROOT }}/,CEPHROOT)} 1' \
    "${DOCKER_IMAGE_PATH}/.datajoint_config.json" > ${BUILDER_HOME}/.datajoint_config.json

SHELL ["/bin/bash", "--login", "-c"]
ENTRYPOINT ["conda-run"]
CMD ["tail", "-f", "/dev/null"]
