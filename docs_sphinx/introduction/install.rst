Installation
============

.. contents::
    :local:
    :depth: 1

There are various ways to install Brian, and we recommend that you chose the installation method that
they are most familiar with and use for other Python packages. If you do not yet have Python installed on
your system (in particular on Windows machines), you can install Python and all of Brian's dependencies
via the `Anaconda distribution <https://www.anaconda.com/distribution/#download-section>`_. You can then install
Brian with the ``conda`` package manager as detailed below.

.. note::
    You need to have access to Python >=3.12 (see Brian's :ref:`support policy <supported_python>`). In particular,
    Brian no longer supports Python 2 (the last version to support Python 2 was :ref:`brian2.3`). All provided
    Python packages also require a 64 bit system, but every desktop or laptop machine built in the last 10 years (and
    even most older machines) is 64 bit compatible.

If you are relying on Python packages for several, independent projects, we recommend that you make use
of separate environments for each project. In this way, you can safely update and install packages for
one of your projects without affecting the others. Both, ``conda`` and ``pip`` support installation in
environments -- for more explanations see the respective instructions below.

Standard install
----------------

.. tabs::

    .. group-tab:: conda package

       We recommend installing Brian into a separate environment, see
       `conda's documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
       for more details.
       Brian 2 is not part of the main Anaconda distribution, but built using the community-maintained
       `conda-forge <https://conda-forge.org/>`_ project. You will therefore have to to install it from the
       `conda-forge channel <https://anaconda.org/conda-forge>`_. To do so, use::

         conda install -c conda-forge brian2

       You can also permanently add the channel to your list of channels::

         conda config --add channels conda-forge

       This has only to be done once. After that, you can install and update the brian2 packages as any other
       Anaconda package::

         conda install brian2


    .. group-tab:: PyPI package (``pip``)

       We recommend installing Brian into a separate "virtual environment", see the
       `Python Packaging User Guide <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/>`_
       for more information.
       Brian is included in the PyPI package index: https://pypi.python.org/pypi/Brian2
       You can therefore install it with the ``pip`` utility::

         python -m pip install brian2

       In rare cases where your current environment does not have access to the ``pip`` utility, you first
       have to install ``pip`` via::

         python -m ensurepip

    .. group-tab:: Ubuntu/Debian package

       If you are using a recent `Debian <https://debian.org>`_-based Linux distribution (Debian itself, or one if its
       derivatives like `Ubuntu <https://ubuntu.com>`_ or `Linux Mint <https://linuxmint.com/>`_), you can install Brian
       using its built-in package manager::

         sudo apt install python3-brian

       Brian releases get packaged by the `Debian Med <https://www.debian.org/devel/debian-med/>`_ team, but note that
       it might take a while until the most recent version shows up in the repository.

    .. group-tab:: Fedora package

       If you are using `Fedora Linux <https://getfedora.org/>`_, you can install Brian using its built-in package
       manager::

        sudo dnf install python-brian2

       Brian releases get packaged by the `NeuroFedora <https://docs.fedoraproject.org/en-US/neurofedora/overview/>`_ team,
       but note that it might take a while until the most recent version shows up in the repository.

    .. group-tab:: Spack package

       `Spack <https://spack.io>`_ is a flexible package manager supporting multiple versions, configurations, platforms, and compilers.

       After setting up Spack you can install Brian with the following command::

         spack install py-brian2

.. _updating_install:

Updating an existing installation
---------------------------------
How to update Brian to a new version depends on the installation method you used
previously. Typically, you can run the same command that you used for installation
(sometimes with an additional option to enforce an upgrade, if available):

.. tabs::

  .. group-tab:: conda package

    Depending on whether you added the ``conda-forge`` channel to the list of channels
    or not (see above), you either have to include it in the update command again or
    can leave it away. I.e. use::

      conda update -c conda-forge brian2

    if you did not add the channel, or::

      conda update brian2

    if you did.

  .. group-tab:: PyPI package (``pip``)

    Use the install command together with the ``--upgrade`` or ``-U`` option::

      python -m pip install -U brian2

  .. group-tab:: Ubuntu/Debian package

    Update the package repository and ask for an install. Note that the package will
    also be updated automatically with commands like ``sudo apt full-upgrade``::

      sudo apt update
      sudo apt install python3-brian

  .. group-tab:: Fedora package

    Update the package repository (not necessary in general, since it will be updated
    regularly without asking for it), and ask for an update. Note that the package will
    also be updated automatically with commands like ``sudo dnf upgrade``::

      sudo dnf check-update python-brian2
      sudo dnf upgrade python-brian2


.. _installation_cpp:

Requirements for C++ code generation
------------------------------------

C++ code generation is highly recommended since it can drastically increase the
speed of simulations (see :doc:`../user/computation` for details). To use it,
you need a C++ compiler and Cython_ (automatically installed as a dependency
of Brian).

.. tabs::

   .. tab:: Linux and OS X

      On Linux and Mac OS X, the conda package will automatically install a C++ compiler.
      But even if you install Brian in a different way, you will most likely already have a
      working C++ compiler installed on your system (try calling ``g++ --version``
      in a terminal). If not, use your distribution's package manager to install a ``g++`` package.

   .. tab:: Windows

      On Windows, :ref:`runtime` (i.e. Cython) requires the Visual Studio compiler, but you do not need a full Visual
      Studio installation, installing the much smaller "Build Tools" package is sufficient:

      * Install the `Microsoft Build Tools for Visual Studio <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_.
      * In Build tools, install C++ build tools and ensure the latest versions of MSVCv... build tools and Windows 10 SDK are checked.
      * Make sure that your ``setuptools`` package has at least version 34.4.0 (use ``conda update setuptools`` when
        using Anaconda, or ``python -m pip install --upgrade setuptools`` when using pip).

      For :ref:`cpp_standalone`, you can either use the compiler installed above or any other version of Visual Studio.

Try running the test suite (see :ref:`testing_brian` below) after the installation to make sure everything is working as expected.

.. _development_install:

Development install
-------------------
When you encounter a problem in Brian, we will sometimes ask you to install Brian's latest development version,
which includes changes that were included after its last release.

We regularly upload the latest development version of Brian to PyPI's test server. You can install it via::

    python -m pip install --upgrade --pre -i https://test.pypi.org/simple/ Brian2

Note that this requires that you already have all of Brian's dependencies installed.

If you have ``git`` installed, you can also install directly from github::

    python -m pip install git+https://github.com/brian-team/brian2.git

Finally, in particular if you want to either contribute to Brian's development or regularly test
its latest development version, you can directly clone the git repository at github
(https://github.com/brian-team/brian2) and then run ``pip install -e .``, to install
Brian in "development mode". With this installation, updating the git repository is in
general enough to keep up with changes in the code, i.e. it is not necessary to install
it again.

.. _testing_brian:

Installing other useful packages
--------------------------------
There are various packages that are useful but not necessary for working with
Brian. These include: matplotlib_ (for plotting), pytest_ (for running the test
suite), ipython_ and jupyter_-notebook (for an interactive console).

.. tabs::
    .. group-tab:: conda package

       ::

         conda install matplotlib pytest ipython notebook

    .. group-tab:: PyPI package (``pip``)

       ::

         python -m pip install matplotlib pytest ipython notebook

You should also have a look at the brian2tools_ package, which contains several
useful functions to visualize Brian 2 simulations and recordings.

.. tabs::
    .. group-tab:: conda package

       As of now, ``brian2tools`` is not yet included in the ``conda-forge``
       channel, you therefore have to install it from our own ``brian-team`` channel::

         conda install -c brian-team brian2tools

    .. group-tab:: PyPI package (``pip``)

       ::

         python -m pip install brian2tools


Testing Brian
-------------

If you have the pytest_ testing utility installed, you can run Brian's test
suite::

    import brian2
    brian2.test()

It should end with "OK", showing a number of skipped tests but no errors or
failures. For more control about the tests that are run see the
:doc:`developer documentation on testing <../developer/guidelines/testing>`.

.. _matplotlib: http://matplotlib.org/
.. _ipython: http://ipython.org/
.. _jupyter: http://jupyter.org/
.. _brian2tools: https://brian2tools.readthedocs.io
.. _azure: https://azure.microsoft.com/en-us/services/devops/pipelines/
.. _pytest: https://docs.pytest.org/en/stable/
.. _Cython: http://cython.org/
