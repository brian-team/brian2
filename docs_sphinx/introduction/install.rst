Installation
============

Brian2 is available on the Python package index: https://pypi.python.org/pypi/Brian2

It can therefore be installed using ``easy_install`` or ``pip``::

    easy_install brian2
    pip install brian2

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