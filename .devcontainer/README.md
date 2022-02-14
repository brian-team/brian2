VS Code Development Container
=============================

Using this development environment requires:
* [Docker](https://www.docker.com/get-started)
* [VS Code](https://code.visualstudio.com/)
* [Remote development extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)

Cloning the Brian repository and opening it in VS Code should result in a prompt to reopen the project in a container. Once the container is built, there will be another prompt saying that [`pylance`](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) has been installed and asking if you wish to reload the container to use it (click `Yes` to enable Python language support). You now have an isolated development environment (which will not conflict with packages installed elsewhere on your system) with all the dependencies need for Brian already installed. Further documentation for development in containers can be found here: https://code.visualstudio.com/docs/remote/containers.

The exact dependency versions used in this container will be saved in `.devcontainer/frozen_dependencies.txt`, which may be useful for debugging. 
