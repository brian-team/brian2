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

"Missing compiler_cxx fix for MSVCCompiler"
-------------------------------------------

If you keep seeing this message, do not worry. It's not possible for us to
hide it, but doesn't indicate any problems.

Problems with numerical integration
-----------------------------------

If the beginning of a run takes a long time, the reason might be the automatic
determination of a suitable numerical integration algorithm.
This can in particular happen for complicated equations where sympy's solvers
take a long time trying to solve the equations symbolically (typically failing
in the end). We try to improve this situation (see #351) but until then, chose
a numerical integration algorithm explicitly (:ref:`numerical_integration`).

Integer division
----------------

If you are using integer division in your models for negative numbers, the results
will be different if using the numpy target to any of the other codegen targets.
This is because in Python, ``-1/2==-1`` whereas in C, ``-1/2==0``. This will be
fixed in a future release.