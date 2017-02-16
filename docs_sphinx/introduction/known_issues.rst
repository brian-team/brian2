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

Parallel Brian simulations with the ``weave`` code generation target
--------------------------------------------------------------------

When using the ``weave`` code generation target (the default runtime target on
Python 2.x, see :ref:`runtime` for details), you should avoid running multiple
Brian simulations in parallel. The ``weave`` package caches compiled files,
but this cache is not prepared for multiple concurrent updates. If two Python
scripts (or two processes started from the same Python script, e.g. via the
`multiprocessing` package) try to store compilation results at the same time,
``weave`` will crash with an error message. The ``numpy`` and ``cython`` targets
are not affected by this problem.

Slow standalone simulations
---------------------------

Some versions of the GNU standard library (in particular those used by recent
Ubuntu versions) have a bug that can dramatically slow down simulations in
C++ standalone mode on modern hardware (see #803). As a workaround, Brian will
set an environment variable ``LD_BIND_NOW`` during the execution of standalone
stimulations which changes the way the library is linked so that it does not
suffer from this problem. If this environment variable leads to unwanted
behaviour on your machine, change the
`prefs.devices.cpp_standalone.run_environment_variables` preference.
