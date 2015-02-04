Installation
============

Released version
----------------

.. warning::

    Installing for Windows can be a little more involved: see the section below.

Brian is available on the Python package index: https://pypi.python.org/pypi/brian2

It can therefore be installed using ``easy_install`` or ``pip`` (in newer
versions, this needs the ``--pre`` flag to allow for the installation of a
prelease)::

    easy_install brian2
    pip install --pre brian2  # newer versions of pip
    pip install brian2  # older versions of pip

.. note::

   In principle, the above commands also install Brian's dependencies.
   Unfortunately, however, there are two circumstances where this doesn't work:

   1. If numpy isn't installed yet, it has to be installed in a separate step
      before any other dependencies (``easy_install numpy`` or
      ``pip install numpy``)
   2. On Python 2.x, installing sympy via ``easy_install`` does not work because
      it tries to install the Python 3 version (see this issue_), use ``pip``
      instead.

.. _issue: https://github.com/sympy/sympy/issues/6610

Alternatively, you can download the source package directly and uncompress it.
You can then either run ``python setup.py install`` to install it, or simply add
the source directory to your ``PYTHONPATH``. Note that if you are using
Python 3, directly running from the source directory is not possible, you have
to install the package first.

Windows
-------

We recommend first installing the
`Anaconda distribution <https://store.continuum.io/cshop/anaconda/>`_ of Python 2.7.

In order to get support for compiled C++ code (highly recommended for speed), you need to next
install Visual Studio 2008 Express, which can be freely downloaded from Microsoft
`here <http://go.microsoft.com/?linkid=7729279>`_. If you installed the 64 bit version of Python,
then you now need to modify
the file ``C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\vcvarsall.bat`` according to
`this patch <https://gist.githubusercontent.com/mikofski/11024332/raw/vcvarsall.bat.patch>`_ to
correct a bug in the free versions of Visual Studio.

Annoyingly, if you wish to use standalone mode on Windows, you will also need a more
recent version of Visual Studio: any version of Visual Studio express from
2010 onwards (also
`freely available <http://www.visualstudio.com/en-us/products/visual-studio-express-vs.aspx>`_
from Microsoft). You will need to apply a similar patch to the above.

For further reading on these issues which might help to get compiled C++ code running with a
different Python installation (particularly for 64 bit Windows), see the following:

* http://poquitopicante.blogspot.co.uk/2014/04/fix-vcvarsallbat-to-install-python-27.html
* http://bugs.python.org/issue7511

Development version
-------------------

To run the latest development code, clone the git repository at github:
https://github.com/brian-team/brian2

You can also use ``pip`` to install directly from github, use::

    pip install https://github.com/brian-team/brian2/archive/master.zip


C extensions
------------

During installation, Brian will try to compile a C++ version of the
`~brian2.synapses.spikequeue.SpikeQueue`, which increases the speed of synaptic
propagation. If compilation fails, the pure Python version is used instead.
Note that if you use the source package directly without an install, you have to
trigger this compilation explicitly using
``python setup.py build_ext --inplace``. When you are using the sources from
github, this process additionally needs a working installation of Cython_.

Testing Brian
-------------

If you have the nose_ testing utility installed, you can run Brian's test
suite::

    import brian2
    brian2.test()

It should end with "OK", possibly showing a number of skipped tests but no
warnings or errors. For more control about the tests that are run see the
:doc:`developer documentation on testing <../developer/guidelines/testing>`.

.. _nose: https://pypi.python.org/pypi/nose
.. _Cython: http://cython.org/