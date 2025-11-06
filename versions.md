Package versions
================

This file contains a list of other files where versions of packages are specified so that they can easily be found when upgrading a dependency version, keeping them all in sync.

In the future it would be advantageous to implement an automated way of keep versions synchronised across files e.g. https://github.com/pre-commit/pre-commit/issues/945#issuecomment-527603460 or preferably parsing `.pre-commit-config.yaml` and using it to `pip install` requirements (see discussion here: https://github.com/brian-team/brian2/pull/1449#issuecomment-1372476018). Until then, the files are listed below for manual checking and updating.

* [`README.md`](https://github.com/brian-team/brian2/blob/master/README.md)
* [`rtd-requirements.txt`](https://github.com/brian-team/brian2/blob/master/rtd-requirements.txt)
* [`pyproject.toml`](https://github.com/brian-team/brian2/blob/master/pyproject.toml)
* [`.pre-commit-config.yaml`](https://github.com/brian-team/brian2/blob/master/.pre-commit-config.yaml)
* [`docs_sphinx/conf.py`](https://github.com/brian-team/brian2/blob/master/docs_sphinx/conf.py)
* [`conda-forge/brian2-feedstock/recipe/meta.yaml`](https://github.com/conda-forge/brian2-feedstock/blob/main/recipe/meta.yaml)
* [`.devcontainer/dev-requirements.txt`](https://github.com/brian-team/brian2/blob/master/.devcontainer/dev-requirements.txt)
* [`.devcontainer/devcontainer.json`](https://github.com/brian-team/brian2/blob/master/.devcontainer/devcontainer.json)
* [`.devcontainer/Dockerfile`](https://github.com/brian-team/brian2/blob/master/.devcontainer/Dockerfile)
* [`docker/Dockerfile`](https://github.com/brian-team/brian2/blob/master/docker/Dockerfile)
