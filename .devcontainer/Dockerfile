# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.187.0/containers/ubuntu/.devcontainer/base.Dockerfile

# [Choice] Ubuntu version: bionic, focal
ARG VARIANT="focal"
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

ENV CONDA_DIR=/app/conda \
    SHELL=/bin/bash
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet "https://github.com/conda-forge/miniforge/releases/download/4.10.3-3/Mambaforge-Linux-x86_64.sh" && \
    ${SHELL} "Mambaforge-Linux-x86_64.sh" -f -b -p ${CONDA_DIR} && \
    rm "Mambaforge-Linux-x86_64.sh" && \
    mamba config --system --set auto_update_conda false && \
    mamba config --system --set show_channel_urls true && \
    # mamba install --quiet --yes 'pip' && \
    mamba update --all --quiet --yes && \
    mamba clean --all -f -y && \
    rm -rf $HOME/.cache/yarn

COPY requirements.txt /tmp/requirements.txt

RUN mamba install --quiet --yes --file /tmp/requirements.txt && \
    mamba clean --all -f -y && \
    rm -rf $HOME/.cache/yarn && \
    rm -rf /tmp/requirements.txt
