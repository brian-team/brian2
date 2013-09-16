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
* Elementary functions: ``sqrt``, ``exp``, ``log``, ``log10``, ``abs``
* Trigonometric functions: ``sin``, ``cos``, ``tan``, ``sinh``, ``cosh``,
  ``tanh``, ``arcsin``, ``arccos``, ``arctan``
* General utility functions: ``clip``, ``floor``, ``ceil``

Brian also provides a special purpose function ``int_``, which can be used to
convert a boolean expression or variable into an integer value of 0 or 1. This
is useful to have a conditional evaluation as part of an equation or statement.
This sometimes allows to circumvent the lack of an ``if`` statement. For
example, the following reset statement resets the variable `v` to either `v_r1`
or `v_r2`, depending on the value of `w`:
``'v = v_r1 * int_(w <= 0.5) + v_r2 * int_(w > 0.5)'``

User-provided functions
-----------------------

Python code generation
~~~~~~~~~~~~~~~~~~~~~~
If a function is only used in contexts that use Python code generation,
preparing a function for use with Brian only means specifying its units. The
simplest way to do this is to use the `check_units` decorator::

    @check_units(x1=meter, y1=meter, x2=meter, y2=meter, result=meter)
    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

Another option is to wrap the function in a `Function` object::

    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    # wrap the distance function
    distance = Function(distance, arg_units=[meter, meter, meter, meter],
                        return_unit=meter)

The use of Brian's unit system has the benefit of checking the consistency of
units for every operation but at the expense of performance.
Consider the following function, for example::

    @check_units(I=amp, result=Hz)
    def piecewise_linear(I):
        return np.clip((I-1*nA) * 50*Hz/nA, 0*Hz, 100*Hz)

When Brian runs a simulation, the state variables are stored and passed around
without units for performance reasons. If the above function is used, however,
Brian adds units to its input argument so that the operations inside the
function do not fail with dimension mismatches. Accordingly, units are removed
from the return value so that the function output can be used with the rest
of the code. For better performance, Brian can alter the namespace of the
function when it is executed as part of the simulation and remove all the
units, then pass values without units to the function. In the above example,
this means making the symbol ``nA`` refer to ``1e-9`` and ``Hz`` to ``1``. To
use this mechanism, add the decorator `make_functions` with the
``discard_units`` keyword::

    @make_function(discard_units=True)
    @check_units(I=amp, result=Hz)
    def piecewise_linear(I):
        return np.clip((I-1*nA) * 50*Hz/nA, 0*Hz, 100*Hz)

Note that the use of the function *outside of simulation runs* is not affected,
i.e. using ``piecewise_linear`` still requires a current in Ampere and returns
a rate in Hertz. The ``discard_units`` mechanism does not work in all cases,
e.g. it does not work if the function refers to units as ``brian2.nA`` instead
of ``nA``, if it uses imports inside the function (e.g.
``from brian2 import nA``), etc. The ``discard_units`` can also be switched on
for all functions without having to use the `make_function` decorator by
setting the `codegen.runtime.numpy.discard_units` preference.

Other code generation targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To make a function available for other code generation targets (e.g. C++),
implementations for these targets have to be added. This can be achieved using
the `make_function` decorator. The form of the code (e.g. a simple string or
a dictionary of strings) necessary is target-dependent. An implementation for
the C++ target could look like this::

    @make_function(codes={'cpp':
                         {'support_code':'''
                         double piecewise_linear(I) {
                            if (I < 1e-9)
                                return 0;
                            if (I > 3e-9)
                                return 100;
                            return (I/1e-9 - 1) * 50;
                         }
                         '''
                         }
                         })
    @check_units(I=amp, result=Hz)
    def piecewise_linear(I):
        return np.clip((I-1*nA) * 50*Hz/nA, 0*Hz, 100*Hz)

Alternatively, `FunctionImplementation` objects can be added to the `Function`
object. For a more complex example that also makes the function contribute
additional values to the namespace of a `CodeObject` see `TimedArray`.