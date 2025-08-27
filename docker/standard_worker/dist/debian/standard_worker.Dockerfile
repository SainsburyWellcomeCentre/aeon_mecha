ARG PY_VER
FROM jupyter/docker-stacks-foundation:python-${PY_VER}

USER root
RUN apt update && \
    apt install -y curl ssh git && \
    pip install --upgrade pip && \
    pip install gateway_provisioners && \
    jupyter image-bootstrap install --languages python && \
    chown jovyan:users /usr/local/bin/bootstrap-kernel.sh && \
    chmod 0755 /usr/local/bin/bootstrap-kernel.sh && \
    chown -R jovyan:users /usr/local/bin/kernel-launchers
CMD /usr/local/bin/bootstrap-kernel.sh


# Additional packages
RUN apt install -y graphviz libsm6 libxext6 libgl1 libegl1-mesa

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

USER jovyan
ARG DEPLOY_KEY
COPY --chown=jovyan $DEPLOY_KEY $HOME/.ssh/id_ed25519
RUN chmod u=r,g-rwx,o-rwx $HOME/.ssh/id_ed25519 && \
    ssh-keyscan github.com >> $HOME/.ssh/known_hosts

ARG REPO_OWNER
ARG REPO_NAME
ARG REPO_BRANCH
WORKDIR $HOME
RUN git clone -b ${REPO_BRANCH} git@github.com:${REPO_OWNER}/${REPO_NAME}.git

# Install the repo via pip for codebook image
RUN pip install ./${REPO_NAME}

WORKDIR $HOME/${REPO_NAME}
RUN uv sync
