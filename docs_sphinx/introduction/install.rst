Installation
============

We recommend users to use the `Anaconda distribution <https://store.continuum.io/cshop/anaconda/>`_
by Continuum Analytics. Its use will make the installation of Brian 2 and its
dependencies simpler, since packages are provided in binary form, meaning that
they don't have to be build from the source code at your machine. Furthermore,
our automatic testing on the continuous integration services travis_ and appveyor_
are based on Anaconda, we are therefore confident that it works under this
configuration.

However, Brian 2 can also be installed independent of Anaconda, either with
other Python distributions (`Enthought Canopy <https://www.enthought.com/products/canopy/>`_,
`Python(x,y) for Windows <https://code.google.com/p/pythonxy/>`_, ...) or simply
based on Python and ``pip`` (see :ref:`installation_from_source` below).

Installation with Anaconda
--------------------------

Installing Anaconda
~~~~~~~~~~~~~~~~~~~
`Download the Anaconda distribution <http://continuum.io/downloads>`_
for your Operating System. For Windows users that want to use Python 3.x, we
strongly recommend installing the 32 Bit version even on 64 Bit systems, since
setting the compilation environment (see :ref:`installation_cpp` below) is less
complicated in that case. Note that the choice between Python 2.7 and Python 3.x
is not very important at this stage, Anaconda allows you to create a Python 3
environment from Python 2 Anaconda and vice versa.

After the installation, make sure that your environment is configured to use
the Anaconda distribution. You should have access to the ``conda`` command in
a terminal and running ``python`` (e.g. from your IDE) should show a header like
this, indicating that you are using Anaconda's Python interpreter::

    Python 2.7.10 |Anaconda 2.3.0 (64-bit)| (default, May 28 2015, 17:02:03)
    [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    Anaconda is brought to you by Continuum Analytics.
    Please check out: http://continuum.io/thanks and https://binstar.org

Here's some documentation on how to set up some popular IDEs for Anaconda:
http://docs.continuum.io/anaconda/ide_integration.html

Installing Brian 2
~~~~~~~~~~~~~~~~~~
You can either install Brian 2 in the Anaconda root environment, or create a
new environment for Brian 2 (http://conda.pydata.org/docs/using/envs.html). The
latter has the advantage that you can update (or not update) the dependencies
of Brian 2 independently from the rest of your system.

Since Brian 2 is not part of the main Anaconda distribution, you have to install
it from the `brian-team channel <https://conda.binstar.org/brian-team>`_. To do
so, use::

    conda install -c brian-team brian2

You can also permanently add the channel to your list of channels::

    conda config --add channels brian-team

This has only to be done once. After that, you can install and update the brian2
packages as any other Anaconda package::

    conda install brian2


Installing other useful packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are various packages that are useful but not necessary for working with
Brian. These include: matplotlib_ (for plotting), nose_ (for running the test
suite), ipython_ and ipython-notebook (for an interactive console). To install
them, simply do::

    conda install matplotlib nose ipython ipython-notebook


.. _installation_from_source:

Installation from source
------------------------
If you decide not to use Anaconda, you can install Brian 2 from the Python
package index: https://pypi.python.org/pypi/brian2

To do so, use the ``pip`` utility. Newer versions of ``pip`` require you to use
the ``--pre`` option to install Brian 2 since it is not yet a final release::

    pip install --pre brian2

You might want to add the ``--user`` flag, to install Brian 2 for the local user
only, which means that you don't need administrator privileges for the
installation.

In principle, the above command also install Brian's dependencies.
Unfortunately, this does not work for ``numpy``, it has to be installed in a
separate step before all other dependencies (``pip install numpy``), if it is
not already installed.

If you have an older version of pip, first update pip itself::

    # On Linux/MacOsX:
    pip install -U pip

    # On Windows
    python -m pip install -U pip

If you don't have ``pip`` but you have the ``easy_install`` utility, you can use
it to install ``pip``::

    easy_install pip

If you have neither ``pip`` nor ``easy_install``, use the approach described
here to install ``pip``: https://pip.pypa.io/en/latest/installing.htm


Alternatively, you can download the source package directly and uncompress it.
You can then either run ``python setup.py install`` or
``python setup.py develop`` to install it, or simply add
the source directory to your ``PYTHONPATH`` (this will only work for Python
2.x).


.. _installation_cpp:

Requirements for C++ code generation
------------------------------------

C++ code generation is highly recommended since it can drastically increase the
speed of simulations (see :doc:`../user/computation` for details). To use it,
you need a C++ compiler and either Cython_ or weave_ (only for Python 2.x).
Cython/weave will be automatically installed if you perform the installation via
Anaconda, as recommended. Otherwise you can install them in the usual way, e.g.
using ``pip install cython`` or ``pip install weave``.

On Linux and Mac OS X, you will most likely already have a working C++ compiler
installed (try calling ``g++ --version`` in a terminal). If not, use your
distribution's package manager to install a ``g++`` package, or install ``gcc``
via Anaconda (``conda install gcc``).

On Windows, the necessary steps depend on the Python version you are using:

**Python 2.7**

* Download and install the `Microsoft Visual C++ Compiler for Python 2.7  <http://www.microsoft.com/en-us/download/details.aspx?id=44266>`_

This should be all you need.

**Python 3.4**

* Download and install the `Microsoft .NET Framework 4 <https://www.microsoft.com/en-us/download/details.aspx?id=17851>`_
* Download and install the `Microsoft Windows SDK for Windows 7 and .NET Framework 4 <http://www.microsoft.com/en-in/download/details.aspx?id=8279>`_

For 64 Bit Windows (and Python 3.4), you have to additionally set up your
environment correctly every time you run your Brian script (this is why we
recommend against using this combination on Windows). To do this, run the
following commands (assuming the default installation path) at the CMD prompt,
or put them in a batch file::

    setlocal EnableDelayedExpansion
    CALL "C:\Program Files\Microsoft SDKs\Windows\v7.1\Bin\SetEnv.cmd" /x64 /release
    set DISTUTILS_USE_SDK=1

Try running the test suite (see :ref:`testing_brian` below) after the
installation to make sure everything is working as expected.

Development version
-------------------

To run the latest development code, you can install from brian-team's "dev"
channel with Anaconda. Note that if you previously added the ``brian-team``
channel to your list of channels, you have to first remove it::

    conda config --remove channels brian-team -f

Also uninstall any version of Brian 2 that you might have previously installed::

    conda remove brian2

Finally, install the ``brian2`` package from the development channel::

    conda install -c brian-team/channel/dev brian2

If this fails with an error message about the ``py-cpuinfo`` package (a
dependency that we provide in the main brian-team channel), install it
from the main channel::

    conda install -c brian-team py-cpuinfo

Then repeat the command to install Brian 2 from the development channel.

You can also directly clone the git repository at github
(https://github.com/brian-team/brian2) and then run ``python setup.py install``
or ``python setup.py develop`` or simply add the source directory to your
``PYTHONPATH`` (this will only work for Python 2.x).

Finally, another option is to use ``pip`` to directly install from github::

    pip install https://github.com/brian-team/brian2/archive/master.zip


.. _testing_brian:

Testing Brian
-------------

If you have the nose_ testing utility installed, you can run Brian's test
suite::

    import brian2
    brian2.test()

It should end with "OK", possibly showing a number of skipped tests but no
warnings or errors. For more control about the tests that are run see the
:doc:`developer documentation on testing <../developer/guidelines/testing>`.

.. _matplotlib: http://matplotlib.org/
.. _ipython: http://ipython.org/
.. _travis: https://travis-ci.org/brian-team/brian2
.. _appveyor: https://ci.appveyor.com/project/brianteam/brian2
.. _nose: https://pypi.python.org/pypi/nose
.. _Cython: http://cython.org/
.. _weave: https://github.com/scipy/weave
