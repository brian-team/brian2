Instructions
============

To build the image manually for testing on a local host for the native architecture:

`docker build -t briansimulator/brian -f docker/Dockerfile .`

Alternatively, to test multi-architecture builds and push to docker hub, first login:

`docker login`

Then execute:

`docker buildx build --push --platform linux/amd64,linux/arm64/v8 -o type=image -t briansimulator/brian -f docker/Dockerfile .`

Finally, to run the image:

`docker run -it --init -p 8888:8888 briansimulator/brian`

Or if you prefer a simple `bash` terminal rather than JupyterLab:

`docker run -it --init briansimulator/brian /bin/bash`

Or to run the tests:

`docker run -it --init --rm briansimulator/brian python -c 'import brian2; brian2.test(test_standalone="cpp_standalone")'`
