Known issues
============

.. contents:: List of known issues
    :local:

Cannot find msvcr90d.dll
------------------------

If you see this message coming up, find the file
``PythonDir\Lib\site-packages\numpy\distutils\mingw32ccompiler.py``
and modify the line ``msvcr_dbg_success = build_msvcr_library(debug=True)`` to read
``msvcr_dbg_success = False`` (you can comment out the existing line and add the new line
immediately after).

Problems with numerical integration
-----------------------------------

If the beginning of a run takes a long time, the reason might be the automatic
determination of a suitable numerical integration algorithm.
This can in particular happen for complicated equations where sympy's solvers
take a long time trying to solve the equations symbolically (typically failing
in the end). We try to improve this situation (see #351) but until then, chose
a numerical integration algorithm explicitly (:ref:`numerical_integration`).

Cannot find vcvarsall.bat on standard search path
-------------------------------------------------

If you have Windows and Microsoft Visual Studio 2015 installed, and you are
trying to run with OpenMP support, you will see this error. There is a bug in
OpenMP support for this version of Visual Studio. You can either install a
different version or switch OpenMP off.
