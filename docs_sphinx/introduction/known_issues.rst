Known issues
============

In addition to the issues noted below, you can refer to our
`bug tracker on GitHub <https://github.com/brian-team/brian2/issues?q=is%3Aopen+is%3Aissue+label%3Abug>`__.

.. contents:: List of known issues
    :local:

Cannot find msvcr90d.dll
------------------------

If you see this message coming up, find the file
``PythonDir\Lib\site-packages\numpy\distutils\mingw32ccompiler.py``
and modify the line ``msvcr_dbg_success = build_msvcr_library(debug=True)`` to read
``msvcr_dbg_success = False`` (you can comment out the existing line and add the new line
immediately after).

"AttributeError: MSVCCompiler instance has no attribute 'compiler_cxx'"
-----------------------------------------------------------------------

This is caused by a bug in some versions of numpy on Windows. The easiest solution is to
update to the latest version of numpy.

If that isn't possible, a hacky solution is to modify the numpy code directly to fix the
problem. The following change may work.
Modify line 388 of ``numpy/distutils/ccompiler.py`` from ``elif not self.compiler_cxx:`` to
``elif not hasattr(self, 'compiler_cxx') or not self.compiler_cxx:``. If the line
number is different, it should be nearby. Search for ``elif not self.compiler_cxx`` in
that file.

"Missing compiler_cxx fix for MSVCCompiler"
-------------------------------------------

If you keep seeing this message, do not worry. It's not possible for us to
hide it, but doesn't indicate any problems.

Problems with numerical integration
-----------------------------------

In some cases, the automatic choice of numerical integration method will not be
appropriate, because of a choice of parameters that couldn't be determined in
advance. In this case, typically you will get nan (not a number) values in the
results, or large oscillations. In this case, Brian will generate a warning to
let you know, but will not raise an error.

Jupyter notebooks and C++ standalone mode progress reporting
------------------------------------------------------------

When you run simulations in C++ standalone mode and enable progress reporting
(e.g. by using ``report='text'`` as a keyword argument), the progress will not
be displayed in the jupyter notebook. If you started the notebook from a
terminal, you will find the output there. Unfortunately, this is a tricky
problem to solve at the moment, due to the details of how the jupyter notebook
handles output.

Parallel Brian simulations with C++ standalone
----------------------------------------------

Simulations using the C++ standalone device will create code and store results
in a dedicated directory (``output``, by default). If you run multiple
simulations in parallel, you have to take care that these simulations do not
use the same directory – otherwise, everything from compilation errors to
incorrect results can happen. Either chose a different directory name for each
simulation and provide it as the ``directory`` argument to the
`.set_device` or `~.Device.build` call, or use ``directory=None`` which
will use a randomly chosen unique temporary directory (in ``/tmp`` on
Unix-based systems) for each simulation. If you need to know the directory name,
you can access it after the simulation run via ``device.project_dir``.

.. _parallel_cython:

Parallel Brian simulations with Cython on machines with NFS (e.g. a computing cluster)
--------------------------------------------------------------------------------------

Generated Cython code is stored in a cache directory on disk so that it can be reused
when it is needed again, without recompiling it. Multiple simulations running in
parallel could interfere during the compilation process by trying to generate the same
file at the same time. To avoid this, Brian uses a file locking mechanism that ensures
that only a process at a time can access these files. Unfortunately, this file locking
mechanism is very slow on machines using the Network File System
(`NFS <https://en.wikipedia.org/wiki/Network_File_System>`_), which is often the case on
computing clusters. On such machines, it is recommend to use an independent cache
directory per process, and to disable the file locking mechanism. This can be done with
the following code that has to be run at the beginning of each process::

    from brian2 import *
    import os
    cache_dir = os.path.expanduser(f'~/.cython/brian-pid-{os.getpid()}')
    prefs.codegen.runtime.cython.cache_dir = cache_dir
    prefs.codegen.runtime.cython.multiprocess_safe = False


Slow C++ standalone simulations
-------------------------------

Some versions of the GNU standard library (in particular those used by recent
Ubuntu versions) have a bug that can dramatically slow down simulations in
C++ standalone mode on modern hardware (see :issue:`803`). As a workaround, Brian will
set an environment variable ``LD_BIND_NOW`` during the execution of standalone
simulations which changes the way the library is linked so that it does not
suffer from this problem. If this environment variable leads to unwanted
behaviour on your machine, change the
`prefs.devices.cpp_standalone.run_environment_variables` preference.

Cython fails with compilation error on OS X: ``error: use of undeclared identifier 'isinf'``
--------------------------------------------------------------------------------------------

Try setting the environment variable ``MACOSX_DEPLOYMENT_TARGET=10.9``.

CMD windows open when running Brian on Windows with the Spyder 3 IDE
--------------------------------------------------------------------

This is due to the interaction with the integrated ipython terminal. Either change the
run configuration to "Execute in an external system terminal" or patch the internal
Python function used to spawn processes as described in github issue :issue:`1140`.
