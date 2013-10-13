Installation
============

Brian2 is available on the Python package index: https://pypi.python.org/pypi/Brian2

It can therefore be installed using ``easy_install`` or ``pip`` (this needs the
``--pre`` flag to allow for the installation of a prelease)::

    easy_install brian2
    pip install --pre brian2

.. note::

   In principle, the above commands also install Brian's dependencies.
   Unfortunately, however, there are two circumstances where this doesn't work:

   1. If numpy isn't installed yet, it has to be installed in a separate step
      before any other dependencies (``easy_install numpy`` or
      ``pip install numpy``)
   2. On Python 2.x, installing sympy via ``easy_install`` does not work because
      it tries to install the Python 3 version (see this issue_), use ``pip``
      instead.

.. _issue: http://code.google.com/p/sympy/issues/detail?id=3511

Alternatively, you can download the source package directly and uncompress it.
You can then either run ``python setup.py install`` to install it, or simply add
the source directory to your ``PYTHONPATH``. Note that if you are using
Python 3, directly running from the source directory is not possible, you have
to install the package first.

To run the latest development code, clone the git repository at github:
https://github.com/brian-team/brian2

Testing Brian
-------------

If you have the nose_ testing utility installed, you can run Brian2's test
suite::

    import brian2
    brian2.test()

It should end with "OK", possibly showing a number of skipped tests but no
warnings or errors.

.. _nose: https://pypi.python.org/pypi/nose