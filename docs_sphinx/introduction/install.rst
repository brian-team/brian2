Installation
============

.. contents::
    :local:
    :depth: 1

We recommend users to use the `Anaconda distribution <https://www.anaconda.com/distribution/#download-section>`_
by Continuum Analytics. Its use will make the installation of Brian 2 and its
dependencies simpler, since packages are provided in binary form, meaning that
they don't have to be build from the source code at your machine. Furthermore,
our automatic testing on the continuous integration services travis_ and appveyor_
are based on Anaconda, we are therefore confident that it works under this
configuration.

However, Brian 2 can also be installed independent of Anaconda, either with
other Python distributions (`Enthought Canopy <https://www.enthought.com/products/canopy/>`_,
`Python(x,y) for Windows <http://python-xy.github.io>`_, ...) or simply
based on Python and ``pip`` (see :ref:`installation_from_source` below).

Installation with Anaconda
--------------------------

Installing Anaconda
~~~~~~~~~~~~~~~~~~~
`Download the Anaconda distribution <https://www.anaconda.com/distribution/#download-section>`_
for your Operating System. Note that the choice between Python 2.7 and Python
3.x is not very important at this stage, Anaconda allows you to create a Python
3 environment from Python 2 Anaconda and vice versa.

After the installation, make sure that your environment is configured to use
the Anaconda distribution. You should have access to the ``conda`` command in
a terminal and running ``python`` (e.g. from your IDE) should show a header like
this, indicating that you are using Anaconda's Python interpreter::

    Python 2.7.10 |Anaconda 2.3.0 (64-bit)| (default, May 28 2015, 17:02:03)
    [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux2
    Type "help", "copyright", "credits" or "license" for more information.

Here's some documentation on how to set up some popular IDEs for Anaconda:
https://docs.anaconda.com/anaconda/user-guide/tasks/integration

Installing Brian 2
~~~~~~~~~~~~~~~~~~
.. note::
    The provided Brian 2 packages are only for 64bit systems. If you want to
    install Brian 2 in a 32bit environment, please use the
    :ref:`installation_from_source` instead.

You can either install Brian 2 in the Anaconda root environment, or create a
new environment for Brian 2 (https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
The latter has the advantage that you can update (or not update) the dependencies
of Brian 2 independently from the rest of your system.

Brian 2 is not part of the main Anaconda distribution, but built using the
community-maintained `conda-forge <https://conda-forge.org/>`_ project. You
will therefore have to to install it from the
`conda-forge channel <https://anaconda.org/conda-forge>`_. To do so, use::

    conda install -c conda-forge brian2

You can also permanently add the channel to your list of channels::

    conda config --add channels conda-forge

This has only to be done once. After that, you can install and update the brian2
packages as any other Anaconda package::

    conda install brian2


Installing other useful packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are various packages that are useful but not necessary for working with
Brian. These include: matplotlib_ (for plotting), nose_ (for running the test
suite), ipython_ and jupyter_-notebook (for an interactive console). To install
them from anaconda, simply do::

    conda install matplotlib nose ipython notebook

You should also have a look at the brian2tools_ package, which contains several
useful functions to visualize Brian 2 simulations and recordings. You can
install it with pip or anaconda, similar to Brian 2 itself (but as of now, it is
not included in the ``conda-forge`` channel, you therefore have to install it
from our own ``brian-team`` channel), e.g. with::

    conda install -c brian-team brian2tools

.. _installation_from_source:

Installation with pip
---------------------
If you decide not to use Anaconda, you can install Brian 2 from the Python
package index: https://pypi.python.org/pypi/Brian2

To do so, use the ``pip`` utility::

    pip install brian2

You might want to add the ``--user`` flag, to install Brian 2 for the local user
only, which means that you don't need administrator privileges for the
installation.

If you have an older version of pip, first update pip itself::

    # On Linux/MacOsX:
    pip install -U pip

    # On Windows
    python -m pip install -U pip

If you don't have ``pip`` but you have the ``easy_install`` utility, you can use
it to install ``pip``::

    easy_install pip

If you have neither ``pip`` nor ``easy_install``, use the approach described
here to install ``pip``: https://pip.pypa.io/en/latest/installing/


.. _installation_cpp:

Requirements for C++ code generation
------------------------------------

C++ code generation is highly recommended since it can drastically increase the
speed of simulations (see :doc:`../user/computation` for details). To use it,
you need a C++ compiler and either Cython_ or weave_ (only available for Python 2.x).
Cython/weave will be automatically installed if you perform the installation via
Anaconda, as recommended. Otherwise you can install them in the usual way, e.g.
using ``pip install cython`` or ``pip install weave``.

Linux and OS X
~~~~~~~~~~~~~~
On Linux and Mac OS X, the conda package will automatically install a C++ compiler.
But even if you install Brian from source, you will most likely already have a
working C++ compiler installed on your system (try calling ``g++ --version``
in a terminal). If not, use your distribution's package manager to install a
``g++`` package.

.. _compiler_setup_windows:

Windows
~~~~~~~
On Windows, the necessary steps to get :ref:`runtime` (i.e. Cython/weave) to work
depend on the Python version you are using (also see the
`notes in the Python wiki <https://wiki.python.org/moin/WindowsCompilers#Compilers_Installation_and_configuration>`_):

* **Python >= 3.5**
    * Install the `Microsoft Build Tools for Visual Studio 2017 <https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017>`_.
    * Make sure that your ``setuptools`` package has at least version 34.4.0 (use ``conda update setuptools`` when using Anaconda, or
        ``pip install --upgrade setuptools`` when using pip).

* **Python 2.7**
    * Download and install the `Microsoft Visual C++ Compiler for Python 2.7  <http://www.microsoft.com/en-us/download/details.aspx?id=44266>`_

For :ref:`cpp_standalone`, you can either use the compiler installed above or any other version of Visual Studio -- in this
case, the Python version does not matter.

Try running the test suite (see :ref:`testing_brian` below) after the
installation to make sure everything is working as expected.

Development version
-------------------

To run the latest development code, you can directly clone the git repository at github
(https://github.com/brian-team/brian2) and then run ``pip install -e .``, to install
Brian in "development mode". With this installation, updating the git repository is in
general enough to keep up with changes in the code, i.e. it is not necessary to install
it again.

Another option is to use ``pip`` to directly install from github::

    pip install https://github.com/brian-team/brian2/archive/master.zip


.. _testing_brian:

Testing Brian
-------------

If you have the nose_ testing utility installed, you can run Brian's test
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
.. _travis: https://travis-ci.org/brian-team/brian2
.. _appveyor: https://ci.appveyor.com/project/brianteam/brian2
.. _nose: https://pypi.python.org/pypi/nose
.. _Cython: http://cython.org/
.. _weave: https://github.com/scipy/weave
