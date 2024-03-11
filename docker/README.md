# Docker instructions

## Run docker image
To run the docker image, you can use

`docker run -it --init -p 8888:8888 briansimulator/brian`

This will start a JupyterLab instance inside the container, which you can access from your local browser via the last link
printed to the terminal (`http://127.0.0.1:8888/lab?token=…`) 

Or if you prefer a simple `bash` terminal rather than JupyterLab:

`docker run -it --init briansimulator/brian /bin/bash`

Or to run the tests:

`docker run -it --init --rm briansimulator/brian python -c 'import brian2; brian2.test(test_standalone="cpp_standalone")'`

## Build docker image
To build the docker image locally, follow the following instructions:

### Build package wheel
You need to first build the package wheel(s) for linux using [`cibuildwheel`](https://cibuildwheel.readthedocs.io). Install it via
`pip install cibuildwheel` (or use `pipx`)

#### Build wheel for local architecture
Run `cibuildwheel` to build a package for your architecture:
```
export CIBW_BUILD='cp312-manylinux*'
cibuildwheel --platform linux --arch auto64 --output-dir dist
```

#### Build multi-arch wheel
First install `qemu` for cross-compilation, e.g. on Debian/Ubuntu:

`sudo apt install qemu-user-static`

Then, run
```
export CIBW_BUILD='cp312-manylinux*'
cibuildwheel --platform linux --arch auto64,aarch64 --output-dir dist
```

### Build docker image

You can then build the docker image via

`docker build -t briansimulator/brian -f docker/Dockerfile .`

Alternatively, to test multi-architecture builds and push to docker hub, first login:

`docker login`

Create a new builder that uses the docker-container driver (only needed once):
```
docker buildx create \                                                                                                                
  --name container \
  --driver=docker-container \
  default
```
Then execute:

```
docker buildx build \
  --builder=container \
  --push \
  --platform linux/amd64,linux/arm64/v8 \
  -o type=image \
  -t briansimulator/brian \
  -f docker/Dockerfile \
  .
```
