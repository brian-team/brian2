# Brian2

*A clock-driven simulator for spiking neural networks*

Brian is a free, open source simulator for spiking neural networks. It is written in the Python programming language and is available on almost all platforms. We believe that a simulator should not only save the time of processors, but also the time of scientists. Brian is therefore designed to be easy to learn and use, highly flexible and easily extensible.

Please report issues at the github issue tracker (https://github.com/brian-team/brian2/issues) or in the Brian forum (https://brian.discourse.group).

Documentation for Brian2 can be found at http://brian2.readthedocs.org

Brian2 is released under the terms of the [CeCILL 2.1 license](https://opensource.org/licenses/CECILL-2.1).

If you use Brian for your published research, we kindly ask you to cite our article:

> Stimberg, M, Brette, R, Goodman, DFM. “Brian 2, an Intuitive and Efficient Neural Simulator.” eLife 8 (2019): e47314. doi: [10.7554/eLife.47314](https://doi.org/10.7554/eLife.47314).


[![PyPI version](https://img.shields.io/pypi/v/Brian2.svg)](https://pypi.python.org/pypi/Brian2)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/brian2.svg)](https://anaconda.org/conda-forge/brian2)
[![Debian package](https://img.shields.io/debian/v/python3-brian/testing)](https://packages.debian.org/testing/python3-brian)
[![Fedora package](https://img.shields.io/fedora/v/python3-brian2)](https://packages.fedoraproject.org/pkgs/python-brian2/python3-brian2/)
[![Spack](https://img.shields.io/spack/v/py-brian2)](https://packages.spack.io/package.html?name=py-brian2)
[![AUR version](https://img.shields.io/aur/version/python-brian2)](https://aur.archlinux.org/packages/python-brian2)

[![Docker Pulls](https://img.shields.io/docker/pulls/briansimulator/brian)](https://hub.docker.com/r/briansimulator/brian)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.654861.svg)](https://zenodo.org/doi/10.5281/zenodo.654861)
[![Software Heritage (repository)](https://archive.softwareheritage.org/badge/origin/https://github.com/brian-team/brian2/)](https://archive.softwareheritage.org/browse/origin/?origin_url=https://github.com/brian-team/brian2)
[![Software Heritage (release)](https://archive.softwareheritage.org/badge/swh:1:rel:2d4c5c8c8a6d2318332889df93ab74aef53e2c61/)](https://archive.softwareheritage.org/swh:1:rel:2d4c5c8c8a6d2318332889df93ab74aef53e2c61;origin=https://github.com/brian-team/brian2;visit=swh:1:snp:a90ab7416901a9c5cf6f56d68b3455c65d322afc)

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)
[![Discourse topics](https://img.shields.io/discourse/topics?server=https%3A%2F%2Fbrian.discourse.group)](https://brian.discourse.group)
[![Discourse chat](https://img.shields.io/badge/discourse-chat-4EC820?logo=discourse&link=https%3A%2F%2Fbrian.discourse.group%2Fchat)](https://brian.discourse.group/chat)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Quickstart
Try out Brian on the [mybinder](https://mybinder.org/) service:

[![mybinder](https://static.mybinder.org/badge.svg)](https://mybinder.org/v2/gh/brian-team/brian2-binder/master?filepath=index.ipynb)

## Dependencies
The following packages need to be installed to use Brian 2 (cf. [`pyproject.toml`](pyproject.toml)):

* Python >= 3.9
* NumPy >=1.21
* SymPy >= 1.2
* Cython >= 0.29.21
* PyParsing
* Jinja2 >= 2.7
* setuptools >= 61
* py-cpuinfo (only required on Windows)

For full functionality, you might also want to install:

* GSL >=1.16
* SciPy >=0.13.3
* Matplotlib >= 2.0

To build the documentation:

* Sphinx (>=7)

To run the test suite:

* pytest
* pytest-xdist (optional)

## Testing status for master branch

[![Test status on GitHub Actions](https://github.com/brian-team/brian2/actions/workflows/testsuite.yml/badge.svg)](https://github.com/brian-team/brian2/actions/workflows/testsuite.yml)
[![Publish status on GitHub Actions](https://github.com/brian-team/brian2/actions/workflows/publish.yml/badge.svg)](https://github.com/brian-team/brian2/actions/workflows/publish.yml)
[![Test coverage](https://img.shields.io/coveralls/brian-team/brian2/master.svg)](https://coveralls.io/r/brian-team/brian2?branch=master)
[![Documentation Status](https://readthedocs.org/projects/brian2/badge/?version=stable)](https://brian2.readthedocs.io/en/stable/?badge=stable)
