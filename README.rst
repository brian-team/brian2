Brian2
======

*A clock-driven simulator for spiking neural networks*

Brian is a free, open source simulator for spiking neural networks. It is written in the Python programming language and is available on almost all platforms. We believe that a simulator should not only save the time of processors, but also the time of scientists. Brian is therefore designed to be easy to learn and use, highly flexible and easily extensible.

Please report issues at the github issue tracker (https://github.com/brian-team/brian2/issues) or at the brian support mailing list (http://groups.google.com/group/briansupport/)

Documentation for Brian2 can be found at http://brian2.readthedocs.org

Brian2 is released under the terms of the `CeCILL 2.1 license <https://opensource.org/licenses/CECILL-2.1>`_.

If you use Brian for your published research, we kindly ask you to cite our article:

Stimberg, M, Brette, R, Goodman, DFM. “Brian 2, an Intuitive and Efficient Neural Simulator.” eLife 8 (2019): e47314. `doi: 10.7554/eLife.47314 <https://doi.org/10.7554/eLife.47314>`_.



.. image:: https://img.shields.io/pypi/v/Brian2.svg
        :target: https://pypi.python.org/pypi/Brian2

.. image:: https://img.shields.io/conda/vn/conda-forge/brian2.svg
        :target: https://anaconda.org/conda-forge/brian2

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.654861.svg
        :target: https://doi.org/10.5281/zenodo.654861

.. image:: https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg
        :target: code-of-conduct.md
        :alt: Contributor Covenant

.. image:: https://badges.gitter.im/Join%20Chat.svg
   :alt: Join the chat at https://gitter.im/brian-team/brian2
   :target: https://gitter.im/brian-team/brian2?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

Quickstart
----------
Try out Brian on the `mybinder <https://mybinder.org/>`_ service:

.. image:: http://mybinder.org/badge.svg
  :target: http://mybinder.org/v2/gh/brian-team/brian2-binder/master?filepath=index.ipynb

Dependencies
------------
The following packages need to be installed to use Brian 2:

* Python >= 3.5
* NumPy >=1.10
* SymPy >= 1.2
* Cython >= 0.29
* PyParsing
* Jinja2 >= 2.7
* setuptools >= 21
* py-cpuinfo (only required on Windows)

For full functionality, you might also want to install:

* GSL >=1.16
* SciPy >=0.13.3
* Matplotlib >= 2.0

To build the documentation:

* Sphinx (>=1.8)

To run the test suite:

* pytest
* pytest-xdist (optional)

Testing status for master branch
--------------------------------

.. image:: https://img.shields.io/travis/brian-team/brian2/master.svg
  :target: https://travis-ci.org/brian-team/brian2?branch=master

.. image:: https://img.shields.io/appveyor/ci/brianteam/brian2/master.svg
  :target: https://ci.appveyor.com/project/brianteam/brian2/branch/master

.. image:: https://img.shields.io/coveralls/brian-team/brian2/master.svg
  :target: https://coveralls.io/r/brian-team/brian2?branch=master

.. image:: https://readthedocs.org/projects/brian2/badge/?version=stable
  :target: https://brian2.readthedocs.io/en/stable/?badge=stable
  :alt: Documentation Status
