Instructions
============

To build the image manually for testing on a local host for the native architecture:

`docker buildx build -t briansimulator/brian -f docker/Dockerfile .`

Alternatively, to test multi-architeture builds and push to docker hub, use:

`docker buildx build --push --platform linux/amd64,linux/arm64/v8 -o type=image -t briansimulator/brian -f docker/Dockerfile .`

Then to run the image:

`docker run -it --init --rm -p 8888:8888 briansimulator/brian`
