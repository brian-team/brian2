VS Code Development Container
=============================

Using this development environment requires:
* [Docker](https://www.docker.com/get-started)
* [VS Code](https://code.visualstudio.com/)
* [Remote development extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)

Cloning the Brian repository and opening it in VS Code should result in a prompt to reopen the project in a container. Clicking `Yes` will start the build process. This will take a few minutes for the first time but will be faster on subsequent rebuilds. 

Once the container is built, there will be another prompt saying that [`pylance`](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) has been installed and asking if you wish to reload the container to activate it (click `Yes` to enable Python language support). You now have an isolated development environment (which will not conflict with packages installed elsewhere on your system) with all the dependencies needed for Brian already installed. 

The container environment can be customised in many ways, such as with [dotfiles](https://code.visualstudio.com/docs/remote/containers#_personalizing-with-dotfile-repositories) if hosted in a public repository. Further documentation for development in containers can be found here: https://code.visualstudio.com/docs/remote/containers.

The exact dependency versions used in this container will be saved in `.devcontainer/frozen_dependencies.txt`, which may be useful for debugging. 

Plots can be saved directly to disk e.g. as `png` files for viewing. [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) is also provided for running code in notebooks and interactive plotting. The backend must be chosen before importing `matplotlib` with one of the following "magic" commands:

* `%matplotlib inline` - This is the default and will render images as PNGs in the notebook.
* `%matplotlib widget` - This generates an ipywidget that renders plots with interactive controls (e.g. resizing, panning and zooming).

See the [`README`](https://github.com/matplotlib/ipympl) and these [notes](https://github.com/microsoft/vscode-jupyter/wiki/Using-%25matplotlib-widget-instead-of-%25matplotlib-notebook,tk,etc) on using Jupyter within a VS Code container for more information. Extra [settings](https://github.com/microsoft/vscode-jupyter/wiki/IPyWidget-Support-in-VS-Code-Python) are included to automatically obtain the required supporting files.  

TODO
----

* Synchronise minimum package versions in `dev-requirements.txt`
* Pre-install standard brian dependencies in the `Dockerfile` to cache rebuilds
* Investigate `launch.json` settings for [debugging](https://code.visualstudio.com/docs/editor/debugging)
