VS Code Development Container
=============================

Using this development environment requires:
* [Docker](https://www.docker.com/get-started)
* [VS Code](https://code.visualstudio.com/)
* [Remote development extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)

Cloning the Brian repository and opening it in VS Code should result in a prompt to reopen the project in a container. Clicking `Yes` will start the build process. This will take a few minutes for the first time but will be faster on subsequent rebuilds.

Once the container is built, there will be another prompt saying that [`pylance`](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) has been installed and asking if you wish to reload the container to activate it (click `Yes` to enable Python language support). You now have an isolated development environment (which will not conflict with packages installed elsewhere on your system) with all the dependencies needed for Brian already installed.

Developer tasks, debugging and profiling
----------------------------------------

The container installs the Python debugger, Ruff, Jupyter and ``snakeviz``. Shared VS Code tasks and launch configurations live in ``.vscode/``.

**Tests** — Command Palette → **Tasks: Run Test Task** runs `dev/tools/run_tests.py` (the full non-standalone suite, several minutes). **Run Brian tests (numpy)** is a faster variant that only covers the numpy runtime target, which is usually enough while iterating. To debug tests, use **Run and Debug** and choose **Python: Debug pytest (current file)** or **Python: Debug Brian tests**.

**Formatting and linting** — Python files use Ruff on save, matching the ``ruff-check`` / ``ruff-format`` pre-commit hooks. **Tasks: Run Build Task** runs ``pre-commit run --all-files``. **Ruff check** and **Ruff format** are also available if you want to run those steps on their own.

**Profiling** — open a Python file, then run the **Profile current file** task. It writes a ``cProfile`` dump and opens `snakeviz` on port 8080 (forwarded from the container).

The container environment can be customised in many ways, such as with [dotfiles](https://code.visualstudio.com/docs/remote/containers#_personalizing-with-dotfile-repositories) if hosted in a public repository. Further documentation for development in containers can be found here: https://code.visualstudio.com/docs/remote/containers.

The exact dependency versions used in this container will be saved in `.devcontainer/frozen-requirements.txt`, which may be useful for debugging.

Note, when updating the packages in `.devcontainer/dev-requirements.txt`, the versions specified in `.pre-commit-config.yaml` must also be updated to match in order for the pre-commit hooks to work.

Plots can be saved directly to disk e.g. as `png` files for viewing. [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) is also provided for running code in notebooks and interactive plotting. The backend can be chosen with one of the following "magic" commands:

* `%matplotlib inline` - This is the default and will render images as PNGs in the notebook.
* `%matplotlib widget` - This generates an ipywidget that renders plots with interactive controls (e.g. resizing, panning and zooming).

See the [`README`](https://github.com/matplotlib/ipympl) and these [notes](https://github.com/microsoft/vscode-jupyter/wiki/Using-%25matplotlib-widget-instead-of-%25matplotlib-notebook,tk,etc) on using Jupyter within a VS Code container for more information. Extra [settings](https://github.com/microsoft/vscode-jupyter/wiki/IPyWidget-Support-in-VS-Code-Python) are included to automatically obtain the required supporting files.
