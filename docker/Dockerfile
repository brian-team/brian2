# docker buildx build --push --platform linux/amd64,linux/arm64/v8 -o type=image -t briansimulator/brian -f docker/Dockerfile .
# docker run -it --init --rm -p 8888:8888 briansimulator/brian

ARG BASE_IMAGE_TAG=3.12-slim-bookworm
FROM python:${BASE_IMAGE_TAG}

LABEL maintainer="Ben Evans <B.D.Evans@sussex.ac.uk>" \
    org.opencontainers.image.vendor="The Brian Development Team" \
    org.opencontainers.image.licenses="CeCILL 2.1" \
    org.opencontainers.image.title="Brian Docker Image" \
    org.opencontainers.image.description="Docker image for Brian - a free, open source simulator for spiking neural networks" \
    org.opencontainers.image.homepage="https://briansimulator.org" \
    org.opencontainers.image.url="https://hub.docker.com/r/briansimulator/brian" \
    org.opencontainers.image.source="https://github.com/brian-team/brian2" \
    org.opencontainers.image.documentation="https://brian2.readthedocs.io/"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    git \
    libgsl-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels into the image
COPY dist /tmp/dist

# Install Brian2 and recommended packages
RUN python -m pip install --no-cache-dir --only-binary=:all: \
    scipy \
    matplotlib \
    jupyterlab \
    pytest \
    pytest-xdist \
    && python -m pip install /tmp/dist/Brian2*_$(uname -m).whl brian2tools \
    && rm -rf /tmp/dist

# Create a non-root user
ARG USER="monty" \
    GROUP="monty" \
    PASSWORD="monty"
RUN groupadd ${GROUP} && \
    useradd -ms /bin/bash -g ${GROUP} -G sudo ${USER} && \
    echo "${USER}:${PASSWORD}" | chpasswd
ENV HOME="/home/${USER}"
RUN chown -R ${USER}:${USER} ${HOME}
USER ${USER}

# Copy tutorial notebooks and example scripts to home directory
WORKDIR ${HOME}
RUN git clone https://github.com/brian-team/brian2.git \
    && mv brian2/examples examples \
    && mv brian2/tutorials tutorials \
    && chmod -R +x tutorials \
    && chmod -R +x examples \
    && find . -name '*.ipynb' -exec jupyter trust {} \; \
    && rm -rf brian2

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser"]
