import inspect
import types
from collections.abc import Callable, Mapping

import numpy as np
import sympy
from numpy.random import rand, randn
from sympy import Function as sympy_Function
from sympy import S
from sympy.codegen import cfunctions as sympy_cfunctions

import brian2.units.unitsafefunctions as unitsafe
from brian2.core.preferences import prefs
from brian2.core.variables import Constant
from brian2.units.allunits import second
from brian2.units.fundamentalunits import (
    DIMENSIONLESS,
    Quantity,
    fail_for_dimension_mismatch,
    get_dimensions,
    is_dimensionless,
)

__all__ = ["DEFAULT_FUNCTIONS", "Function", "implementation", "declare_types"]


BRIAN_DTYPES = ["boolean", "integer", "float"]
VALID_ARG_TYPES = BRIAN_DTYPES + ["any"]
VALID_RETURN_TYPES = BRIAN_DTYPES + ["highest"]


def declare_types(**types):
    """
    Decorator to declare argument and result types for a function

    Usage is similar to `check_units` except that types must be one of ``{VALID_ARG_TYPES}``
    and the result type must be one of ``{VALID_RETURN_TYPES}``. Unspecified argument
    types are assumed to be ``'all'`` (i.e. anything is permitted), and an unspecified
    result type is assumed to be ``'float'``. Note that the ``'highest'`` option for
    result type will give the highest type of its argument, e.g. if the arguments
    were boolean and integer then the result would be integer, if the arguments were
    integer and float it would be float.
    """

    def annotate_function_with_types(f):
        if hasattr(f, "_orig_arg_names"):
            arg_names = f._orig_arg_names
        else:
            arg_names = f.__code__.co_varnames[0 : f.__code__.co_argcount]
        argtypes = []
        for name in arg_names:
            arg_type = types.get(name, "any")
            if arg_type not in VALID_ARG_TYPES:
                raise ValueError(
                    f"Argument type {arg_type} is not valid, must be one of"
                    f" {VALID_ARG_TYPES}, for argument {name}"
                )
            argtypes.append(arg_type)
        for n in types:
            if n not in arg_names and n != "result":
                raise ValueError(f"Type specified for unknown argument {n}")
        return_type = types.get("result", "float")
        if return_type not in VALID_RETURN_TYPES:
            raise ValueError(
                f"Result type {return_type} is not valid, must be one of"
                f" {VALID_RETURN_TYPES}"
            )
        f._arg_types = argtypes
        f._return_type = return_type
        f._orig_arg_names = arg_names
        f._annotation_attributes = getattr(f, "_annotation_attributes", []) + [
            "_arg_types",
            "_return_type",
        ]
        return f

    return annotate_function_with_types


class Function:
    """
    An abstract specification of a function that can be used as part of
    model equations, etc.

    Parameters
    ----------
    pyfunc : function
        A Python function that is represented by this `Function` object.
    sympy_func : `sympy.Function`, optional
        A corresponding sympy function (if any). Allows functions to be
        interpreted by sympy and potentially make simplifications. For example,
        ``sqrt(x**2)`` could be replaced by ``abs(x)``.
    arg_units : list of `Unit`, optional
        If `pyfunc` does not provide unit information (which typically means
        that it was not annotated with a `check_units` decorator), the
        units of the arguments have to specified explicitly using this
        parameter.
    return_unit : `Unit` or callable, optional
        Same as for `arg_units`: if `pyfunc` does not provide unit information,
        this information has to be provided explictly here. `return_unit` can
        either be a specific `Unit`, if the function always returns the same
        unit, or a function of the input units, e.g. a "square" function would
        return the square of its input units, i.e. `return_unit` could be
        specified as ``lambda u: u**2``.
    arg_types : list of str, optional
        Similar to `arg_units`, but gives the type of the argument rather than
        its unit. In the current version of Brian arguments are specified
        by one of the following strings: 'boolean', 'integer', 'float', 'any'.
        If `arg_types` is not specified, 'any' will be assumed. In
        future versions, a more refined specification may be possible. Note that
        any argument with a type other than float should have no units. If
    return_type : str, optional
        Similar to `return_unit` and `arg_types`. In addition to 'boolean',
        'integer' and 'float' you can also use 'highest' which will return the
        highest type of its arguments. You can also give a function, as for
        `return_unit`. If the return type is not specified, it is assumed to
        be 'float'.
    stateless : bool, optional
        Whether this function does not have an internal state, i.e. if it
        always returns the same output when called with the same arguments.
        This is true for mathematical functions but not true for ``rand()``, for
        example. Defaults to ``True``.
    auto_vectorise : bool, optional
        Whether the implementations of this function should get an additional
        argument (not specified in abstract code) that can be used to determine
        the number of values that should be returned (for the numpy target), or
        an index potentially useful for generating deterministic values
        independent of the order of vectorisation (for all other targets). The
        main use case are random number functions, e.g. equations refer to
        ``rand()``, but the generate code will actually call
        ``rand(_vectorisation_idx)``. Defaults to ``False``.

    Notes
    -----
    If a function should be usable for code generation targets other than
    Python/numpy, implementations for these target languages have to be added
    using the `~brian2.codegen.functions.implementation` decorator or using the
    `~brian2.codegen.functions.add_implementations` function.
    """

    def __init__(
        self,
        pyfunc,
        sympy_func=None,
        arg_units=None,
        arg_names=None,
        return_unit=None,
        arg_types=None,
        return_type=None,
        stateless=True,
        auto_vectorise=False,
    ):
        self.pyfunc = pyfunc
        self.sympy_func = sympy_func
        self._arg_units = arg_units
        self._arg_names = arg_names
        self._return_unit = return_unit
        if return_unit == bool:
            self._returns_bool = True
        else:
            self._returns_bool = False
        self._arg_types = arg_types
        self._return_type = return_type
        self.stateless = stateless
        self.auto_vectorise = auto_vectorise
        if self._arg_units is None:
            if not hasattr(pyfunc, "_arg_units"):
                raise ValueError(
                    f"The Python function '{pyfunc.__name__}' does not specify "
                    "how it deals with units, need to specify "
                    "'arg_units' or use the '@check_units' "
                    "decorator."
                )
            elif pyfunc._arg_units is None:
                # @check_units sets _arg_units to None if the units aren't
                # specified for all of its arguments
                raise ValueError(
                    f"The Python function '{pyfunc.__name__}' does not "
                    "specify the units for all of its arguments."
                )
            else:
                self._arg_units = pyfunc._arg_units
        else:
            if any(isinstance(u, str) for u in self._arg_units):
                if self._arg_names is None:
                    raise TypeError("Need to specify the names of the arguments.")
                if len(self._arg_names) != len(self._arg_units):
                    raise TypeError(
                        "arg_names and arg_units need to have the "
                        f"same length ({len(self._arg_names)} != "
                        f"({len(self._arg_units)})"
                    )

        if self._return_unit is None:
            if not hasattr(pyfunc, "_return_unit"):
                raise ValueError(
                    f"The Python function '{pyfunc.__name__}' does not "
                    "specify how it deals with units, need to specify "
                    "'return_unit' or use the '@check_units' decorator."
                )
            elif pyfunc._return_unit is None:
                # @check_units sets _return_unit to None if no "result=..."
                # keyword is specified.
                raise ValueError(
                    f"The Python function '{pyfunc.__name__}' does not "
                    "specify the unit for its return value."
                )
            else:
                self._return_unit = pyfunc._return_unit

        if self._arg_types is None:
            if hasattr(pyfunc, "_arg_types"):
                self._arg_types = pyfunc._arg_types
            else:
                self._arg_types = ["any"] * len(self._arg_units)

        if self._return_type is None:
            self._return_type = getattr(pyfunc, "_return_type", "float")

        for argtype, u in zip(self._arg_types, self._arg_units):
            if (
                argtype != "float"
                and argtype != "any"
                and u is not None
                and not is_dimensionless(u)
            ):
                raise TypeError(
                    "Non-float arguments must be dimensionless in function"
                    f" {pyfunc.__name__}"
                )
            if argtype not in VALID_ARG_TYPES:
                raise ValueError(
                    f"Argument type {argtype} is not valid, must be one "
                    f"of {VALID_ARG_TYPES}, in function "
                    f"'{pyfunc.__name__}'."
                )

        if self._return_type not in VALID_RETURN_TYPES:
            raise ValueError(
                f"Return type {self._return_typ} is not valid, must "
                f"be one of {VALID_RETURN_TYPES}, in function "
                f"'{pyfunc.__name__}'"
            )

        #: Stores implementations for this function in a
        #: `FunctionImplementationContainer`
        self.implementations = FunctionImplementationContainer(self)

    def is_locally_constant(self, dt):
        """
        Return whether this function (if interpreted as a function of time)
        should be considered constant over a timestep. This is most importantly
        used by `TimedArray` so that linear integration can be used. In its
        standard implementation, always returns ``False``.

        Parameters
        ----------
        dt : float
            The length of a timestep (without units).

        Returns
        -------
        constant : bool
            Whether the results of this function can be considered constant
            over one timestep of length `dt`.
        """
        return False

    def __call__(self, *args):
        return self.pyfunc(*args)


class FunctionImplementation:
    """
    A simple container object for function implementations.

    Parameters
    ----------
    name : str, optional
        The name of the function in the target language. Should only be
        specified if the function has to be renamed for the target language.
    code : language-dependent, optional
        A language dependent argument specifying the implementation in the
        target language, e.g. a code string or a dictionary of code strings.
    namespace : dict-like, optional
        A dictionary of mappings from names to values that should be added
        to the namespace of a `CodeObject` using the function.
    dependencies : dict-like, optional
        A mapping of names to `Function` objects, for additional functions
        needed by this function.
    availability_check : callable, optional
        A function that will be called to check whether the function should be
        made available (e.g. depending on whether it is supported by the
        compiler). The function should do nothing if the function is
        available, or raise a ``NotImplementedError`` with a message
        explaining why it isn't.
    dynamic : bool, optional
        Whether this `code`/`namespace` is dynamic, i.e. generated for each
        new context it is used in. If set to ``True``, `code` and `namespace`
        have to be callable with a `Group` as an argument and are expected
        to return the final `code` and `namespace`. Defaults to ``False``.
    """

    def __init__(
        self,
        name=None,
        code=None,
        namespace=None,
        dependencies=None,
        availability_check=None,
        dynamic=False,
        compiler_kwds=None,
    ):
        if compiler_kwds is None:
            compiler_kwds = {}
        self.name = name
        if dependencies is None:
            dependencies = {}
        self.dependencies = dependencies
        self._code = code
        self._namespace = namespace
        self.dynamic = dynamic
        self.compiler_kwds = compiler_kwds
        self.availability_check = availability_check

    def get_code(self, owner):
        if self.availability_check is not None:
            self.availability_check()
        if self.dynamic:
            return self._code(owner)
        else:
            return self._code

    def get_namespace(self, owner):
        if self.dynamic:
            return self._namespace(owner)
        else:
            return self._namespace


class FunctionImplementationContainer(Mapping):
    """
    Helper object to store implementations and give access in a dictionary-like
    fashion, using `CodeGenerator` implementations as a fallback for `CodeObject`
    implementations.
    """

    def __init__(self, function):
        self._function = function
        self._implementations = dict()

    def __getitem__(self, key):
        """
        Find an implementation for this function that can be used by the
        `CodeObject` given as `key`. Will find implementations registered
        for `key` itself (or one of its parents), or for the `CodeGenerator`
        class that `key` uses (or one of its parents). In all cases,
        implementations registered for the corresponding names qualify as well.

        Parameters
        ----------
        key : `CodeObject`
            The `CodeObject` that will use the `Function`

        Returns
        -------
        implementation : `FunctionImplementation`
            An implementation suitable for `key`.
        """
        fallback = getattr(key, "generator_class", None)
        # in some cases we do the code generation with original_generator_class instead (e.g. GSL)
        fallback_parent = getattr(key, "original_generator_class", None)

        for K in [key, fallback, fallback_parent]:
            name = getattr(K, "class_name", "no class name for key")
            for impl_key, impl in self._implementations.items():
                impl_key_name = getattr(
                    impl_key, "class_name", "no class name for implementation"
                )
                if (impl_key_name is not None and impl_key_name in [K, name]) or (
                    impl_key is not None and impl_key in [K, name]
                ):
                    return impl
            if hasattr(K, "__bases__"):
                for cls in inspect.getmro(K):
                    if cls in self._implementations:
                        return self._implementations[cls]
                    name = getattr(cls, "class_name", None)
                    if name in self._implementations:
                        return self._implementations[name]

        # Give a nicer error message if possible
        if getattr(key, "class_name", None) is not None:
            key = key.class_name
        elif getattr(fallback, "class_name", None) is not None:
            key = fallback.class_name
        keys = ", ".join(
            [getattr(k, "class_name", str(k)) for k in self._implementations]
        )
        raise KeyError(
            f"No implementation available for target '{key}'. "
            f"Available implementations: {keys}"
        )

    def add_numpy_implementation(
        self, wrapped_func, dependencies=None, discard_units=None, compiler_kwds=None
    ):
        """
        Add a numpy implementation to a `Function`.

        Parameters
        ----------
        function : `Function`
            The function description for which an implementation should be added.
        wrapped_func : callable
            The original function (that will be used for the numpy implementation)
        dependencies : list of `Function`, optional
            A list of functions this function needs.
        discard_units : bool, optional
            See `implementation`.
        """
        if discard_units is None:
            discard_units = prefs["codegen.runtime.numpy.discard_units"]

        # Get the original function inside the check_units decorator
        if hasattr(wrapped_func, "_orig_func"):
            orig_func = wrapped_func._orig_func
        else:
            orig_func = wrapped_func

        if discard_units:
            new_globals = dict(orig_func.__globals__)
            # strip away units in the function by changing its namespace
            for key, value in new_globals.items():
                if isinstance(value, Quantity):
                    new_globals[key] = np.asarray(value)
            unitless_func = types.FunctionType(
                orig_func.__code__,
                new_globals,
                orig_func.__name__,
                orig_func.__defaults__,
                orig_func.__closure__,
            )
            self._implementations["numpy"] = FunctionImplementation(
                name=None,
                code=unitless_func,
                dependencies=dependencies,
                compiler_kwds=None,
            )
        else:

            def wrapper_function(*args):
                arg_units = list(self._function._arg_units)

                if self._function.auto_vectorise:
                    arg_units += [DIMENSIONLESS]
                if not len(args) == len(arg_units):
                    func_name = self._function.pyfunc.__name__
                    raise ValueError(
                        f"Function {func_name} got {len(args)} arguments, "
                        f"expected {len(arg_units)}."
                    )
                new_args = []
                for arg, arg_unit in zip(args, arg_units):
                    if (
                        arg_unit == bool
                        or arg_unit is None
                        or isinstance(arg_unit, str)
                    ):
                        new_args.append(arg)
                    else:
                        new_args.append(
                            Quantity.with_dimensions(arg, get_dimensions(arg_unit))
                        )
                result = orig_func(*new_args)
                if isinstance(self._function._return_unit, Callable):
                    return_unit = self._function._return_unit(
                        *[get_dimensions(a) for a in args]
                    )
                else:
                    return_unit = self._function._return_unit
                if return_unit == bool:
                    if not (
                        isinstance(result, bool) or np.asarray(result).dtype == bool
                    ):
                        raise TypeError(
                            f"The function {orig_func.__name__} returned "
                            f"'{result}', but it was expected to return a "
                            "boolean value "
                        )
                elif (
                    isinstance(return_unit, int) and return_unit == 1
                ) or return_unit.dim is DIMENSIONLESS:
                    fail_for_dimension_mismatch(
                        result,
                        return_unit,
                        f"The function '{orig_func.__name__}' "
                        f"returned {result}, but it was "
                        "expected to return a dimensionless "
                        "quantity.",
                    )
                else:
                    fail_for_dimension_mismatch(
                        result,
                        return_unit,
                        f"The function '{orig_func.__name__}' "
                        f"returned {result}, but it was "
                        "expected to return a quantity with "
                        f"units {return_unit!r}.",
                    )
                return np.asarray(result)

            self._implementations["numpy"] = FunctionImplementation(
                name=None, code=wrapper_function, dependencies=dependencies
            )

    def add_implementation(
        self,
        target,
        code,
        namespace=None,
        dependencies=None,
        availability_check=None,
        name=None,
        compiler_kwds=None,
    ):
        self._implementations[target] = FunctionImplementation(
            name=name,
            code=code,
            dependencies=dependencies,
            availability_check=availability_check,
            namespace=namespace,
            compiler_kwds=compiler_kwds,
        )

    def add_dynamic_implementation(
        self,
        target,
        code,
        namespace=None,
        dependencies=None,
        availability_check=None,
        name=None,
        compiler_kwds=None,
    ):
        """
        Adds an "dynamic implementation" for this function. `code` and `namespace`
        arguments are expected to be callables that will be called in
        `Network.before_run` with the owner of the `CodeObject` as an argument.
        This allows to generate code that depends on details of the context it
        is run in, e.g. the ``dt`` of a clock.
        """
        if not callable(code):
            raise TypeError(
                f"code argument has to be a callable, is type {type(code)} instead"
            )
        if namespace is not None and not callable(namespace):
            raise TypeError(
                f"namespace argument has to be a callable, is type {type(code)} instead"
            )
        self._implementations[target] = FunctionImplementation(
            name=name,
            code=code,
            namespace=namespace,
            dependencies=dependencies,
            availability_check=availability_check,
            dynamic=True,
            compiler_kwds=compiler_kwds,
        )

    def __len__(self):
        return len(self._implementations)

    def __iter__(self):
        return iter(self._implementations)


def implementation(
    target,
    code=None,
    namespace=None,
    dependencies=None,
    discard_units=None,
    name=None,
    **compiler_kwds,
):
    """
    A simple decorator to extend user-written Python functions to work with code
    generation in other languages.

    Parameters
    ----------
    target : str
        Name of the code generation target (e.g. ``'cython'``) for which to add
        an implementation.
    code : str or dict-like, optional
        What kind of code the target language expects is language-specific,
        e.g. C++ code allows for a dictionary of code blocks instead of a
        single string.
    namespaces : dict-like, optional
        A namespace dictionary (i.e. a mapping of names to values) that
        should be added to a `CodeObject` namespace when using this function.
    dependencies : dict-like, optional
        A mapping of names to `Function` objects, for additional functions
        needed by this function.
    discard_units: bool, optional
        Numpy functions can internally make use of the unit system. However,
        during a simulation run, state variables are passed around as unitless
        values for efficiency. If `discard_units` is set to ``False``, input
        arguments will have units added to them so that the function can still
        use units internally (the units will be stripped away from the return
        value as well). Alternatively, if `discard_units` is set to ``True``,
        the function will receive unitless values as its input. The namespace
        of the function will be altered to make references to units (e.g.
        ``ms``) refer to the corresponding floating point values so that no
        unit mismatch errors are raised. Note that this system cannot work in
        all cases, e.g. it does not work with functions that internally imports
        values (e.g. does ``from brian2 import ms``) or access values with
        units indirectly (e.g. uses ``brian2.ms`` instead of ``ms``). If no
        value is given, defaults to the preference setting
        `codegen.runtime.numpy.discard_units`.
    name : str, optional
        The name of the function in the target language. Should only be
        specified if the function has to be renamed for the target language.
    compiler_kwds : dict, optional
        Additional keyword arguments will be transferred to the code generation
        stage, e.g. for C++-based targets, the code can make use of additional
        header files by providing a list of strings as the ``headers`` argument.

    Notes
    -----
    While it is in principle possible to provide a numpy implementation
    as an argument for this decorator, this is normally not necessary -- the
    numpy implementation should be provided in the decorated function.

    If this decorator is used with other decorators such as `check_units` or
    `declare_types`, it should be the uppermost decorator (that is, the
    last one to be applied).

    Examples
    --------
    Sample usage::

        @implementation('cpp','''
                    #include<math.h>
                    inline double usersin(double x)
                    {
                        return sin(x);
                    }
                    ''')
        def usersin(x):
            return sin(x)
    """

    def do_user_implementation(func):
        # Allow nesting of decorators
        if isinstance(func, Function):
            function = func
        else:
            function = Function(func)

        if discard_units:  # Add a numpy implementation that discards units
            if not (target == "numpy" and code is None):
                raise TypeError(
                    "'discard_units' can only be set for code "
                    "generation target 'numpy', without providing "
                    "any code."
                )
            function.implementations.add_numpy_implementation(
                wrapped_func=func,
                dependencies=dependencies,
                discard_units=discard_units,
                compiler_kwds=compiler_kwds,
            )
        else:
            function.implementations.add_implementation(
                target,
                code=code,
                dependencies=dependencies,
                namespace=namespace,
                name=name,
                compiler_kwds=compiler_kwds,
            )
        # # copy any annotation attributes
        # if hasattr(func, '_annotation_attributes'):
        #     for attrname in func._annotation_attributes:
        #         setattr(function, attrname, getattr(func, attrname))
        # function._annotation_attributes = getattr(func, '_annotation_attributes', [])
        return function

    return do_user_implementation


class SymbolicConstant(Constant):
    """
    Class for representing constants (e.g. pi) that are understood by sympy.
    """

    def __init__(self, name, sympy_obj, value):
        super().__init__(name, value=value)
        self.sympy_obj = sympy_obj


################################################################################
# Standard functions and constants
################################################################################


def _exprel(x):
    if x.is_zero:
        return S.One
    else:
        return (sympy.exp(x) - S.One) / x


class exprel(sympy_Function):
    """
    Represents ``(exp(x) - 1)/x``.

    The benefit of using ``exprel(x)`` over ``(exp(x) - 1)/x``
    is that the latter is prone to cancellation under finite precision
    arithmetic when x is close to zero, and cannot be evaluated when x is
    equal to zero.
    """

    nargs = 1

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return (sympy.exp(*self.args) * (self.args[0] - S.One) + S.One) / self.args[
                0
            ] ** 2
        else:
            raise sympy.ArgumentIndexError(self, argindex)

    def _eval_expand_func(self, **hints):
        return _exprel(*self.args)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        if arg.is_zero:
            return S.One
        else:
            return (sympy.exp(arg) - S.One) / arg

    _eval_rewrite_as_tractable = _eval_rewrite_as_exp

    @classmethod
    def eval(cls, arg):
        if arg is None:
            return None
        if arg.is_zero:
            return S.One

        exp_arg = sympy.exp.eval(arg)
        if exp_arg is not None:
            return (exp_arg - S.One) / arg

    def _eval_is_real(self):
        return self.args[0].is_real

    def _eval_is_finite(self):
        return self.args[0].is_finite


_infinity_int = 1073741823  # maximum 32bit integer divided by 2


def timestep(t, dt):
    """
    Converts a given time to an integer time step. This function slightly shifts
    the time before dividing it by ``dt`` to make sure that multiples of ``dt``
    do not end up in the preceding time step due to floating point issues. This
    function is used in the refractoriness calculation.

    .. versionadded:: 2.1.3

    Parameters
    ----------
    t : np.ndarray, float, Quantity
        The time to convert.
    dt : float or Quantity
        The length of a simulation time step.

    Returns
    -------
    ts : np.ndarray, np.int64
        The time step corresponding to the given time.

    Notes
    -----
    This function cannot handle infinity values, use big values instead (e.g.
    a `NeuronGroup` will use ``-1e4*second`` as the value of the ``lastspike``
    variable for neurons that never spiked).
    """
    elapsed_steps = np.array((t + 1e-3 * dt) / dt, dtype=np.int64)
    if elapsed_steps.shape == ():
        elapsed_steps = elapsed_steps.item()
    return elapsed_steps


DEFAULT_FUNCTIONS = {
    # numpy functions that have the same name in numpy and math.h
    "cos": Function(
        unitsafe.cos, sympy_func=sympy.functions.elementary.trigonometric.cos
    ),
    "sin": Function(
        unitsafe.sin, sympy_func=sympy.functions.elementary.trigonometric.sin
    ),
    "tan": Function(
        unitsafe.tan, sympy_func=sympy.functions.elementary.trigonometric.tan
    ),
    "cosh": Function(
        unitsafe.cosh, sympy_func=sympy.functions.elementary.hyperbolic.cosh
    ),
    "sinh": Function(
        unitsafe.sinh, sympy_func=sympy.functions.elementary.hyperbolic.sinh
    ),
    "tanh": Function(
        unitsafe.tanh, sympy_func=sympy.functions.elementary.hyperbolic.tanh
    ),
    "exp": Function(
        unitsafe.exp, sympy_func=sympy.functions.elementary.exponential.exp
    ),
    "log": Function(
        unitsafe.log, sympy_func=sympy.functions.elementary.exponential.log
    ),
    "log10": Function(unitsafe.log10, sympy_func=sympy_cfunctions.log10),
    "expm1": Function(unitsafe.expm1, sympy_func=sympy_cfunctions.expm1),
    "exprel": Function(unitsafe.exprel, sympy_func=exprel),
    "log1p": Function(unitsafe.log1p, sympy_func=sympy_cfunctions.log1p),
    "sqrt": Function(
        np.sqrt,
        sympy_func=sympy.functions.elementary.miscellaneous.sqrt,
        arg_units=[None],
        return_unit=lambda u: u**0.5,
    ),
    "ceil": Function(
        np.ceil,
        sympy_func=sympy.functions.elementary.integers.ceiling,
        arg_units=[None],
        return_unit=lambda u: u,
    ),
    "floor": Function(
        np.floor,
        sympy_func=sympy.functions.elementary.integers.floor,
        arg_units=[None],
        return_unit=lambda u: u,
    ),
    # numpy functions that have a different name in numpy and math.h
    "arccos": Function(
        unitsafe.arccos, sympy_func=sympy.functions.elementary.trigonometric.acos
    ),
    "arcsin": Function(
        unitsafe.arcsin, sympy_func=sympy.functions.elementary.trigonometric.asin
    ),
    "arctan": Function(
        unitsafe.arctan, sympy_func=sympy.functions.elementary.trigonometric.atan
    ),
    "abs": Function(
        np.abs,
        return_type="highest",
        sympy_func=sympy.functions.elementary.complexes.Abs,
        arg_units=[None],
        return_unit=lambda u: u,
    ),
    "sign": Function(
        pyfunc=np.sign,
        sympy_func=sympy.sign,
        return_type="highest",
        arg_units=[None],
        return_unit=1,
    ),
    # functions that need special treatment
    "rand": Function(
        pyfunc=rand, arg_units=[], return_unit=1, stateless=False, auto_vectorise=True
    ),
    "randn": Function(
        pyfunc=randn, arg_units=[], return_unit=1, stateless=False, auto_vectorise=True
    ),
    "poisson": Function(
        pyfunc=np.random.poisson,
        arg_units=[1],
        return_unit=1,
        return_type="integer",
        stateless=False,
        auto_vectorise=True,
    ),
    "clip": Function(
        pyfunc=np.clip,
        arg_units=[None, "a", "a"],
        arg_names=["a", "a_min", "a_max"],
        return_type="highest",
        return_unit=lambda u1, u2, u3: u1,
    ),
    "int": Function(
        pyfunc=np.int_, return_type="integer", arg_units=[1], return_unit=1
    ),
    "timestep": Function(
        pyfunc=timestep,
        return_type="integer",
        arg_units=[second, second],
        return_unit=1,
    ),
}

DEFAULT_CONSTANTS = {
    "pi": SymbolicConstant("pi", sympy.pi, value=np.pi),
    "e": SymbolicConstant("e", sympy.E, value=np.e),
    "inf": SymbolicConstant("inf", S.Infinity, value=np.inf),
    "-inf": SymbolicConstant("-inf", S.NegativeInfinity, value=-np.inf),
}
