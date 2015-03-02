Computational methods and efficiency
====================================

Brian has several different methods for running the computations in a
simulation. In particular, Brian uses "runtime code generation" for
efficient computation. This means that it takes the Python code and strings
in your model and generates code in one of several possible different
languages and actually executes that. The target language for this code
generation process is set in the `codegen.target` preference. By default, this
preference is set to ``'auto'``, meaning that it will chose a compiled language
target if possible and fall back to Python otherwise. There are two compiled
language targets for Python 2.x, ``'weave'`` (needing a working installation of
a C++ compiler) and ``'cython'`` (needing the `Cython`_ package in addition);
for Python 3.x, only ``'cython'`` is available. If you want to chose a code
generation target explicitly (e.g. because you want to get rid of the warning
that only the Python fallback is available), set the preference to ``'numpy'``,
``'weave'`` or ``'cython'`` at the beginning of your script::

    from brian2 import *
    prefs.codegen.target = 'numpy'  # use the Python fallback

See :doc:`../advanced/preferences` for different ways of setting preferences.
If you are using a compiled language target, also see the
`Compiler settings for maximum speed`_ section below.

 .. _Cython: http://cython.org/

Both of these code generation targets are still run via Python, which means
that there are still overheads due to Python. The fastest way to run
Brian is in "standalone mode" (see :doc:`devices`), although this won't work
for every possible simulation. Note that you can also use multiple threads
with standalone mode, which is not possible in the modes described above.
This doesn't always lead to a huge speed improvement, but can occasionally
give a higher than linear speed up relative to the number of cores.

You might find that running simulations in weave or Cython modes won't work
or is not as efficient as you were expecting. This is probably because you're
using Python functions which are not compatible with weave or Cython. For
example, if you wrote something like this it would not be efficient::

    from brian2 import *
    prefs.codegen.target = 'cython'
    def f(x):
        return abs(x)
    G = NeuronGroup(10000, 'dv/dt = -x*f(x) : 1')
    
The reason is that the function ``f(x)`` is a Python function and so cannot
be called from C++ directly. To solve this problem, you need to provide an
implementation of the function in the target language. See :doc:`functions`.

Compiler settings for maximum speed
-----------------------------------

If using C++ code generation (either via weave, cython or standalone), you
can maximise the efficiency of the generated code in various ways, described
below. These can be set in the global preferences file as described in
:doc:`../advanced/preferences`.

GCC
~~~

For the GCC compiler, the fastest options are::

    codegen.cpp.extra_compile_args_gcc = ['-w', '-Ofast', '-march=native']
    
The ``-Ofast`` optimisation allows the compiler to disregard strict IEEE standards
compliance. In our usage this has never been a problem, but we don't do this
by default for safety. Note that not all versions of gcc include this switch,
older versions might require you to write ``'-O3', '-ffast-math'``.

The ``-march=native`` sets the computer architecture to be the one available
on the machine you are compiling on. This allows the compiler to make use of
as many advanced instructions as possible, but reduces portability of the
generated executable (which is not usually an issue). Again, this option
is not available on all versions of gcc so on an older version you might have
to put your architecture in explicitly (check the gcc docs for your version).

MSVC
~~~~

For the MSVC compiler, the fastest options are::

    codegen.cpp.extra_compile_args_msvc = ['/Ox', '/EHsc', '/w', '/arch:AVX2', '/fp:fast']
    
Note that as above for ``-Ofast`` on gcc, ``/fp:fast`` will enable the
compiler to disregard strict IEEE standards compliance, which has never
been a problem in our usage but we leave this off by default for safety.

The ``/arch:AVX2`` option may not be available on your version of MSVC and
your computer architecture. The available options (in order from best to
worst) are: ``AVX2``, ``AVX``, ``SSE2``, ``SSE`` and ``IA32``.
