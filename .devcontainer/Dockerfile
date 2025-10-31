# https://github.com/devcontainers/images/tree/main/src/python

ARG VARIANT="3.14-bookworm"
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}

# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends libgsl-dev

# [Optional] Uncomment this line to install global node packages.
# RUN su vscode -c "source /usr/local/share/nvm/nvm.sh && npm install -g <your-package-here>" 2>&1

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# Upgrade pip and install developer requirements
COPY .devcontainer/dev-requirements.txt /tmp/pip-tmp/
RUN pip3 install --upgrade pip \
    && pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/dev-requirements.txt \
    && rm -rf /tmp/pip-tmp
