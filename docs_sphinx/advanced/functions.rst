Functions
=========

All equations, expressions and statements in Brian can make use of mathematical
functions. However, functions have to be prepared for use with Brian for two
reasons: 1) Brian is strict about checking the consistency of units, therefore
every function has to specify how it deals with units; 2) functions need to
be implemented differently for different code generation targets.

Brian provides a number of default functions that are already prepared for use
with numpy and C++ and also provides a mechanism for preparing new functions
for use (see below).

Default functions
-----------------
The following functions (stored in the `DEFAULT_FUNCTIONS` dictionary) are
ready for use:

* Random numbers: ``rand()``, ``randn()`` (Note that these functions should be
  called without arguments, the code generation process will take care of
  generating an array of numbers for numpy).
* Elementary functions: ``sqrt``, ``exp``, ``log``, ``log10``, ``abs``, ``sign``
* Trigonometric functions: ``sin``, ``cos``, ``tan``, ``sinh``, ``cosh``,
  ``tanh``, ``arcsin``, ``arccos``, ``arctan``
* General utility functions: ``clip``, ``floor``, ``ceil``

Brian also provides a special purpose function ``int``, which can be used to
convert a an expression or variable into an integer value. This is especially
useful for boolean values (which will be converted into 0 or 1), for example to
have a conditional evaluation as part of an equation or statement which
sometimes allows to circumvent the lack of an ``if`` statement. For
example, the following reset statement resets the variable `v` to either `v_r1`
or `v_r2`, depending on the value of `w`:
``'v = v_r1 * int(w <= 0.5) + v_r2 * int(w > 0.5)'``

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
``'support_code'`` block. Note that both ``'cpp'`` and ``'weave'`` can be used
to provide C++ implementations, the first should be used for generic C++
implementations, and the latter if weave-specific code is necessary. An
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

Arrays vs. scalar values in user-provided functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Equations, expressions and abstract code statements are always implicitly
referring to all the neurons in a `NeuronGroup`, all the synapses in a
`Synapses` object, etc. Therefore, function calls also apply to more than a
single value. The way in which this is handled differs between code generation
targets that support vectorized expressions (e.g. the ``numpy`` target) and
targets that don't (e.g. the ``weave`` target or the ``cpp_standalone`` mode).
If the code generation target supports vectorized expressions, it will receive
an array of values. For example, in the ``piecewise_linear`` example above, the
argument ``I`` will be an array of values and the function returns an array of
values. For code generation without support for vectorized expressions, all
code will be executed in a loop (over neurons, over synapses, ...), the function
will therefore be called several times with a single value each time.

In both cases, the function will only receive the "relevant" values, meaning
that if for example a function is evaluated as part of a reset statement, it
will only receive values for the neurons that just spiked.

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
from weave or Cython code). Scalar values should be directly included in the
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
