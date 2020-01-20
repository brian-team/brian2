Functions
=========

.. contents::
    :local:
    :depth: 2

All equations, expressions and statements in Brian can make use of mathematical
functions. However, functions have to be prepared for use with Brian for two
reasons: 1) Brian is strict about checking the consistency of units, therefore
every function has to specify how it deals with units; 2) functions need to
be implemented differently for different code generation targets.

Brian provides a number of default functions that are already prepared for use
with numpy and C++ and also provides a mechanism for preparing new functions
for use (see below).

.. _default_functions:

Default functions
-----------------
The following functions (stored in the `DEFAULT_FUNCTIONS` dictionary) are
ready for use:

* Random numbers: ``rand`` (random numbers drawn from a uniform distribution
  between 0 and 1), ``randn`` (random numbers drawn from the standard normal
  distribution, i.e. with mean 0 and standard deviation 1),
  and ``poisson`` (discrete random numbers from a Poisson distribution with rate
  parameter :math:`\lambda`)
* Elementary functions: ``sqrt``, ``exp``, ``log``, ``log10``, ``abs``, ``sign``
* Trigonometric functions: ``sin``, ``cos``, ``tan``, ``sinh``, ``cosh``,
  ``tanh``, ``arcsin``, ``arccos``, ``arctan``
* Functions for improved numerical accuracy: ``expm1`` (calculates ``exp(x) - 1``, more accurate
  for ``x`` close to 0), ``log1p`` (calculates ``log(1 + x)``, more accurate for ``x`` close to 0),
  and ``exprel`` (calculates ``(exp(x) - 1)/x``, more accurate for ``x`` close to 0, and returning
  1.0 instead of ``NaN`` for ``x == 0``
* General utility functions: ``clip``, ``floor``, ``ceil``

Brian also provides a special purpose function ``int``, which can be used to
convert an expression or variable into an integer value. This is especially
useful for boolean values (which will be converted into 0 or 1), for example to
have a conditional evaluation as part of an equation or statement which
sometimes allows to circumvent the lack of an ``if`` statement. For
example, the following reset statement resets the variable `v` to either `v_r1`
or `v_r2`, depending on the value of `w`:
``'v = v_r1 * int(w <= 0.5) + v_r2 * int(w > 0.5)'``

Finally, the function `~brian2.core.functions.timestep` is a function that takes
a time and the length of a time step as an input and returns an integer
corresponding to the respective time step. The advantage of using this function
over a simple division is that it slightly shifts the time before dividing to
avoid floating point issues. This function is used as part of the
:doc:`../user/refractoriness` mechanism.

.. _user_functions:

User-provided functions
-----------------------

Python code generation
~~~~~~~~~~~~~~~~~~~~~~
If a function is only used in contexts that use Python code generation,
preparing a function for use with Brian only means specifying its units. The
simplest way to do this is to use the `check_units` decorator::

    @check_units(x1=meter, y1=meter, x2=meter, y2=meter, result=meter)
    def distance(x1, y1, x2, y2):
        return sqrt((x1 - x2)**2 + (y1 - y2)**2)

Another option is to wrap the function in a `Function` object::

    def distance(x1, y1, x2, y2):
        return sqrt((x1 - x2)**2 + (y1 - y2)**2)
    # wrap the distance function
    distance = Function(distance, arg_units=[meter, meter, meter, meter],
                        return_unit=meter)

The use of Brian's unit system has the benefit of checking the consistency of
units for every operation but at the expense of performance.
Consider the following function, for example::

    @check_units(I=amp, result=Hz)
    def piecewise_linear(I):
        return clip((I-1*nA) * 50*Hz/nA, 0*Hz, 100*Hz)

When Brian runs a simulation, the state variables are stored and passed around
without units for performance reasons. If the above function is used, however,
Brian adds units to its input argument so that the operations inside the
function do not fail with dimension mismatches. Accordingly, units are removed
from the return value so that the function output can be used with the rest
of the code. For better performance, Brian can alter the namespace of the
function when it is executed as part of the simulation and remove all the
units, then pass values without units to the function. In the above example,
this means making the symbol ``nA`` refer to ``1e-9`` and ``Hz`` to ``1``. To
use this mechanism, add the decorator `implementation` with the
``discard_units`` keyword::

    @implementation('numpy', discard_units=True)
    @check_units(I=amp, result=Hz)
    def piecewise_linear(I):
        return clip((I-1*nA) * 50*Hz/nA, 0*Hz, 100*Hz)

Note that the use of the function *outside of simulation runs* is not affected,
i.e. using ``piecewise_linear`` still requires a current in Ampere and returns
a rate in Hertz. The ``discard_units`` mechanism does not work in all cases,
e.g. it does not work if the function refers to units as ``brian2.nA`` instead
of ``nA``, if it uses imports inside the function (e.g.
``from brian2 import nA``), etc. The ``discard_units`` can also be switched on
for all functions without having to use the `implementation` decorator by
setting the `codegen.runtime.numpy.discard_units` preference.

Other code generation targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To make a function available for other code generation targets (e.g. C++),
implementations for these targets have to be added. This can be achieved using
the `implementation` decorator. The form of the code (e.g. a simple string or
a dictionary of strings) necessary is target-dependent, for C++ both options
are allowed, a simple string will be interpreted as filling the
``'support_code'`` block. Note that ``'cpp'`` is used
to provide C++ implementations. An
implementation for the C++ target could look like this::

    @implementation('cpp', '''
         double piecewise_linear(double I) {
            if (I < 1e-9)
                return 0;
            if (I > 3e-9)
                return 100;
            return (I/1e-9 - 1) * 50;
         }
         ''')
    @check_units(I=amp, result=Hz)
    def piecewise_linear(I):
        return clip((I-1*nA) * 50*Hz/nA, 0*Hz, 100*Hz)

Alternatively, `FunctionImplementation` objects can be added to the `Function`
object.

The same sort of approach as for C++ works for Cython using the
``'cython'`` target. The example above would look like this::

    @implementation('cython', '''
        cdef double piecewise_linear(double I):
            if I<1e-9:
                return 0.0
            elif I>3e-9:
                return 100.0
            return (I/1e-9-1)*50
        ''')
    @check_units(I=amp, result=Hz)
    def piecewise_linear(I):
        return clip((I-1*nA) * 50*Hz/nA, 0*Hz, 100*Hz)

Dependencies between functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The code generation mechanism for user-defined functions only adds the source
code for a function when it is necessary. If a user-defined function refers to
another function in its source code, it therefore has to explicitly state this
dependency so that the code of the dependency is added as well::

    @implementation('cpp','''
        double rectified_linear(double x)
        {
            return clip(x, 0, INFINITY);
        }''',
        dependencies={'clip': DEFAULT_FUNCTIONS['clip']}
        )
    @check_units(x=1, result=1)
    def rectified_linear(x):
        return np.clip(x, 0, np.inf)

.. note::
    The dependency mechanism is unnecessary for the ``numpy`` code generation
    target, since functions are defined as actual Python functions and not as
    code given in a string.

Additional compiler arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If the code for a function needs additional compiler options to work, e.g. to
link to an external library, these options can be provided as keyword
arguments to the ``@implementation`` decorator. E.g. to link C++ code to the
``foo`` library which is stored in the directory ``/usr/local/foo``, use::

        @implementation('cpp', '...',
         libraries=['foo'], library_dirs=['/usr/local/foo'])

These arguments can also be used to refer to external source files, see
:ref:`below <external_sources>`. Equivalent arguments can also be set as global
:doc:`preferences` in which case they apply to all code and not only to code
referring to the respective function. Note that in C++ standalone mode, all
files are compiled together, and therefore the additional compiler arguments
provided to functions are always combined with the preferences into a common
set of settings that is applied to all code.

The list of currently supported additional arguments (for further explications,
see the respective :doc:`preferences` and the Python documentation of the
`distutils.core.Extension` class):

========================   ============== ======
keyword                    C++ standalone Cython
========================   ============== ======
``headers``                ✓              ❌
``sources``                ✓              ✓
``define_macros``          ✓              ❌
``libraries``              ✓              ✓
``include_dirs``           ✓              ✓
``library_dirs``           ✓              ✓
``runtime_library_dirs``   ✓              ✓
========================   ============== ======

Arrays vs. scalar values in user-provided functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Equations, expressions and abstract code statements are always implicitly
referring to all the neurons in a `NeuronGroup`, all the synapses in a
`Synapses` object, etc. Therefore, function calls also apply to more than a
single value. The way in which this is handled differs between code generation
targets that support vectorized expressions (e.g. the ``numpy`` target) and
targets that don't (e.g. the ``cpp_standalone`` mode).
If the code generation target supports vectorized expressions, it will receive
an array of values. For example, in the ``piecewise_linear`` example above, the
argument ``I`` will be an array of values and the function returns an array of
values. For code generation without support for vectorized expressions, all
code will be executed in a loop (over neurons, over synapses, ...), the function
will therefore be called several times with a single value each time.

In both cases, the function will only receive the "relevant" values, meaning
that if for example a function is evaluated as part of a reset statement, it
will only receive values for the neurons that just spiked.

.. _function_vectorisation:

Functions with context-dependent return values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When using the ``numpy`` target, functions have to return an array of values
(e.g. one value for each neuron). In some cases, the number of values to return
cannot be deduced from the function's arguments. Most importantly, this is the
case for random numbers: a call to `rand()` has to return one value for each
neuron if it is part of a neuron's equations, but only one value for each neuron
that spiked during the time step if it is part of the reset statement. Such
function are said to "auto vectorise", which means that their implementation
receives an additional array argument ``_vectorisation_idx``; the length of this
array determines the number of values the function should return. This argument
is also provided to functions for other code generation targets, but in these
cases it is a single value (e.g. the index of the neuron), and is currently
ignored. To enable this property on a user-defined function, you'll currently
have to manually create a `Function` object::

    def exponential_rand(l, _vectorisation_idx):
        '''Generate a number from an exponential distribution using inverse
           transform sampling'''
        uniform = np.random.rand(len(_vectorisation_idx))
        return -(1/l)*np.log(1 - uniform)

    exponential_rand = Function(exponential_rand, arg_units=[1], return_unit=1,
                                stateless=False, auto_vectorise=True)

Implementations for other code generation targets can then be added using the
`~FunctionImplementationContainer.add_implementation` mechanism::

    cpp_code = '''
    double exponential_rand(double l, int _vectorisation_idx)
    {
        double uniform = rand(_vectorisation_idx);
        return -(1/l)*log(1 - uniform);
    }
    '''
    exponential_rand.implementations.add_implementation('cpp', cpp_code,
                                                        dependencies={'rand': DEFAULT_FUNCTIONS['rand'],
                                                                      'log': DEFAULT_FUNCTIONS['log']})

Note that by referring to the `rand` function, the new random number generator will
automatically generate reproducible random numbers if the `seed` function is use to set
its seed. Restoring the random number state with `restore` will have the expected effect
as well.

Additional namespace
~~~~~~~~~~~~~~~~~~~~
Some functions need additional data to compute a result, e.g. a `TimedArray`
needs access to the underlying array. For the ``numpy`` target, a function can
simply use a reference to an object defined outside the function, there is no
need to explicitly pass values in a namespace. For the other code language
targets, values can be passed in the ``namespace`` argument of the
`implementation` decorator or the
`~brian2.core.functions.FunctionImplementationContainer.add_implementation` method. The namespace
values are then accessible in the function code under the given name, prefixed
with ``_namespace``. Note that this mechanism should only be used for numpy
arrays or general objects (e.g. function references to call Python functions
from Cython code). Scalar values should be directly included in the
function code, by using a "dynamic implemention" (see
`~brian2.core.functions.FunctionImplementationContainer.add_dynamic_implementation`).

See `TimedArray` and `BinomialFunction` for examples that use this mechanism.

Data types
~~~~~~~~~~

By default, functions are assumed to take any type of argument, and return a floating
point value. If you want to put a restriction on the type of an argument, or specify
that the return type should be something other than float, either declare it as a
`Function` (and see its documentation on specifying types) or use the `declare_types`
decorator, e.g.::

    @check_units(a=1, b=1, result=1)
    @declare_types(a='integer', result='highest')
    def f(a, b):
        return a*b

This is potentially important if you have functions that return integer or boolean
values, because Brian's code generation optimisation step will make some potentially
incorrect simplifications if it assumes that the return type is floating point.

.. _external_sources:

External source files
~~~~~~~~~~~~~~~~~~~~~

Code for functions can also be provided via external files in the target
language. This can be especially useful for linking to existing code without
having to include it a second time in the Python script. For C++-based code
generation targets (i.e. the C++ standalone mode), the external
code should be in a file that is provided as an argument to the ``sources``
keyword, together with a header file whose name is provided to ``headers``
(see the note for the `codegen.cpp.headers` preference about the necessary
format). Since the main simulation code is compiled and executed in a different
directory, you should also point the compiler towards the directory of the
header file via the ``include_dirs`` keyword. For the same reason, use an
absolute path for the source file.
For example, the ``piecewise_linear`` function from above can be implemented
with external files as follows:

.. code-block:: cpp

    //file: piecewise_linear.h
    double piecewise_linear(double);

.. code-block:: cpp

    //file: piecewise_linear.cpp
    double piecewise_linear(double I) {
        if (I < 1e-9)
            return 0;
        if (I > 3e-9)
            return 100;
        return (I/1e-9 - 1) * 50;
    }

.. code::

    # Python script

    # Get the absolute directory of this Python script, the C++ files are
    # expected to be stored alongside of it
    import os
    current_dir = os.path.abspath(os.path.dirname(__file__))

    @implementation('cpp', '// all code in piecewise_linear.cpp',
                    sources=[os.path.join(current_dir,
                                          'piecewise_linear.cpp')],
                    headers=['"piecewise_linear.h"'],
                    include_dirs=[current_dir])
    @check_units(I=amp, result=Hz)
    def piecewise_linear(I):
        return clip((I-1*nA) * 50*Hz/nA, 0*Hz, 100*Hz)


For Cython, the process is very similar (see the
`Cython documentation <https://cython.readthedocs.io/en/latest/src/userguide/sharing_declarations.html>`_
for general information). The name of the header file does not need to be
specified, it is expected to have the same name as the source file (except for
the ``.pxd`` extension). The source and header files will be automatically
copied to the cache directory where Cython files are compiled, they therefore
have to be imported as top-level modules, regardless of whether the executed
Python code is itself in a package or module.

A Cython equivalent of above's C++ example can be written as:

.. code-block:: cython

    # file: piecewise_linear.pxd
    cdef double piecewise_linear(double)

.. code-block:: cython

    # file: piecewise_linear.pyx
    cdef double piecewise_linear(double I):
        if I<1e-9:
            return 0.0
        elif I>3e-9:
            return 100.0
        return (I/1e-9-1)*50

.. code::

    # Python script

    # Get the absolute directory of this Python script, the Cython files
    # are expected to be stored alongside of it
    import os
    current_dir = os.path.abspath(os.path.dirname(__file__))

    @implementation('cython',
                    'from piecewise_linear cimport piecewise_linear',
                    sources=[os.path.join(current_dir,
                                          'piecewise_linear.pyx')])
    @check_units(I=amp, result=Hz)
    def piecewise_linear(I):
        return clip((I-1*nA) * 50*Hz/nA, 0*Hz, 100*Hz)
