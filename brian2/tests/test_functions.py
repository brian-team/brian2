import os
import shutil
import tempfile

import pytest
from numpy.testing import assert_equal

from brian2 import *
from brian2.codegen.codeobject import CodeObject
from brian2.codegen.cpp_prefs import compiler_supports_c99
from brian2.codegen.generators import CodeGenerator
from brian2.core.functions import timestep
from brian2.devices import RuntimeDevice
from brian2.parsing.sympytools import str_to_sympy, sympy_to_str
from brian2.tests.utils import assert_allclose, exc_isinstance
from brian2.utils.logger import catch_logs


@pytest.mark.codegen_independent
def test_constants_sympy():
    """
    Make sure that symbolic constants are understood correctly by sympy
    """
    assert sympy_to_str(str_to_sympy("1.0/inf")) == "0"
    assert sympy_to_str(str_to_sympy("sin(pi)")) == "0"
    assert sympy_to_str(str_to_sympy("log(e)")) == "1"


@pytest.mark.standalone_compatible
def test_constants_values():
    """
    Make sure that symbolic constants use the correct values in code
    """
    G = NeuronGroup(3, "v : 1")
    G.v[0] = "pi"
    G.v[1] = "e"
    G.v[2] = "inf"
    run(0 * ms)
    assert_allclose(G.v[:], [np.pi, np.e, np.inf])


def int_(x):
    return array(x, dtype=int)


int_.__name__ = "int"


@pytest.mark.parametrize(
    "func,needs_c99_support",
    [
        (cos, False),
        (tan, False),
        (sinh, False),
        (cosh, False),
        (tanh, False),
        (arcsin, False),
        (arccos, False),
        (arctan, False),
        (log, False),
        (log10, False),
        (log1p, True),
        (exp, False),
        (np.sqrt, False),
        (expm1, True),
        (exprel, True),
        (np.ceil, False),
        (np.floor, False),
        (np.sign, False),
        (int_, False),
    ],
)
@pytest.mark.standalone_compatible
def test_math_functions(func, needs_c99_support):
    """
    Test that math functions give the same result, regardless of whether used
    directly or in generated Python or C++ code.
    """
    if not get_device() == RuntimeDevice or prefs.codegen.target != "numpy":
        if needs_c99_support and not compiler_supports_c99():
            pytest.skip('Support for function "{}" needs a compiler with C99 support.')
    test_array = np.array([-1, -0.5, 0, 0.5, 1])

    with catch_logs() as _:  # Let's suppress warnings about illegal values
        # Calculate the result directly
        numpy_result = func(test_array)

        # Calculate the result in a somewhat complicated way by using a
        # subexpression in a NeuronGroup
        if func.__name__ == "absolute":
            # we want to use the name abs instead of absolute
            func_name = "abs"
        else:
            func_name = func.__name__
        G = NeuronGroup(
            len(test_array),
            f"""func = {func_name}(variable) : 1
                           variable : 1""",
        )
        G.variable = test_array
        mon = StateMonitor(G, "func", record=True)
        net = Network(G, mon)
        net.run(defaultclock.dt)

        assert_allclose(
            numpy_result,
            mon.func_.flatten(),
            err_msg=f"Function {func.__name__} did not return the correct values",
        )


@pytest.mark.standalone_compatible
@pytest.mark.parametrize("func,operator", [(np.power, "**"), (np.mod, "%")])
def test_math_operators(func, operator):
    default_dt = defaultclock.dt
    test_array = np.array([-1, -0.5, 0, 0.5, 1])
    # Functions/operators
    scalar = 3

    # Calculate the result directly
    numpy_result = func(test_array, scalar)

    # Calculate the result in a somewhat complicated way by using a
    # subexpression in a NeuronGroup
    G = NeuronGroup(
        len(test_array),
        f"""func = variable {operator} scalar : 1
                       variable : 1""",
    )
    G.variable = test_array
    mon = StateMonitor(G, "func", record=True)
    net = Network(G, mon)
    net.run(defaultclock.dt)

    assert_allclose(
        numpy_result,
        mon.func_.flatten(),
        err_msg=f"Function {func.__name__} did not return the correct values",
    )


@pytest.mark.standalone_compatible
def test_clip():
    G = NeuronGroup(
        4,
        """
        clipexpr1 = clip(integer_var1, 0, 1) : integer
        clipexpr2 = clip(integer_var2, -0.5, 1.5) : integer
        clipexpr3 = clip(float_var1, 0, 1) : 1
        clipexpr4 = clip(float_var2, -0.5, 1.5) : 1
        integer_var1 : integer
        integer_var2 : integer
        float_var1 : 1
        float_var2 : 1
        """,
    )
    G.integer_var1 = [0, 1, -1, 2]
    G.integer_var2 = [0, 1, -1, 2]
    G.float_var1 = [0.0, 1.0, -1.0, 2.0]
    G.float_var2 = [0.0, 1.0, -1.0, 2.0]
    s_mon = StateMonitor(
        G, ["clipexpr1", "clipexpr2", "clipexpr3", "clipexpr4"], record=True
    )
    run(defaultclock.dt)
    assert_equal(s_mon.clipexpr1.flatten(), [0, 1, 0, 1])
    assert_equal(s_mon.clipexpr2.flatten(), [0, 1, 0, 1])
    assert_allclose(s_mon.clipexpr3.flatten(), [0, 1, 0, 1])
    assert_allclose(s_mon.clipexpr4.flatten(), [0, 1, -0.5, 1.5])


@pytest.mark.standalone_compatible
def test_bool_to_int():
    # Test that boolean expressions and variables are correctly converted into
    # integers
    G = NeuronGroup(
        2,
        """
        intexpr1 = int(bool_var) : integer
        intexpr2 = int(float_var > 1.0) : integer
        bool_var : boolean
        float_var : 1
        """,
    )
    G.bool_var = [True, False]
    G.float_var = [2.0, 0.5]
    s_mon = StateMonitor(G, ["intexpr1", "intexpr2"], record=True)
    run(defaultclock.dt)
    assert_equal(s_mon.intexpr1.flatten(), [1, 0])
    assert_equal(s_mon.intexpr2.flatten(), [1, 0])


@pytest.mark.standalone_compatible
def test_integer_power():
    # See github issue #1500
    G = NeuronGroup(
        3,
        """
        intval1 : integer
        intval2 : integer
        k : integer (constant)
        """,
    )
    G.k = [0, 1, 2]
    G.run_regularly("intval1 = 2**k; intval2 = (-1)**k")
    run(defaultclock.dt)
    assert_equal(G.intval1[:], [1, 2, 4])
    assert_equal(G.intval2[:], [1, -1, 1])


@pytest.mark.codegen_independent
def test_timestep_function():
    dt = defaultclock.dt_
    # Check that multiples of dt end up in the correct time step
    t = np.arange(100000) * dt
    assert_equal(timestep(t, dt), np.arange(100000))

    # Scalar values should stay scalar
    ts = timestep(0.0005, 0.0001)
    assert np.isscalar(ts) and ts == 5

    # Length-1 arrays should stay arrays
    ts = timestep(np.array([0.0005]), 0.0001)
    assert ts.shape == (1,) and ts == 5


@pytest.mark.standalone_compatible
def test_timestep_function_during_run():
    group = NeuronGroup(
        2,
        """ref_t : second
                              ts = timestep(ref_t, dt) + timestep(t, dt) : integer""",
    )
    group.ref_t = [-1e4 * second, 5 * defaultclock.dt]
    mon = StateMonitor(group, "ts", record=True)
    run(5 * defaultclock.dt)
    assert all(mon.ts[0] <= -1e4)
    assert_equal(mon.ts[1], [5, 6, 7, 8, 9])


@pytest.mark.standalone_compatible
def test_user_defined_function():
    @implementation(
        "cpp",
        """
        inline double usersin(double x)
        {
            return sin(x);
        }
        """,
    )
    @implementation(
        "cython",
        """
        cdef double usersin(double x):
            return sin(x)
        """,
    )
    @check_units(x=1, result=1)
    def usersin(x):
        return np.sin(x)

    default_dt = defaultclock.dt
    test_array = np.array([0, 1, 2, 3])
    G = NeuronGroup(
        len(test_array),
        """
        func = usersin(variable) : 1
        variable : 1
        """,
    )
    G.variable = test_array
    mon = StateMonitor(G, "func", record=True)
    run(default_dt)
    assert_allclose(np.sin(test_array), mon.func_.flatten())


def test_user_defined_function_units():
    """
    Test the preparation of functions for use in code with check_units.
    """
    prefs.codegen.target = "numpy"
    if prefs.codegen.target != "numpy":
        pytest.skip("numpy-only test")

    def nothing_specified(x, y, z):
        return x * (y + z)

    no_result_unit = check_units(x=1, y=second, z=second)(nothing_specified)
    all_specified = check_units(x=1, y=second, z=second, result=second)(
        nothing_specified
    )
    consistent_units = check_units(x=None, y=None, z="y", result=lambda x, y, z: x * y)(
        nothing_specified
    )

    G = NeuronGroup(
        1,
        """
        a : 1
        b : second
        c : second
        """,
        namespace={
            "nothing_specified": nothing_specified,
            "no_result_unit": no_result_unit,
            "all_specified": all_specified,
            "consistent_units": consistent_units,
        },
    )
    net = Network(G)
    net.run(0 * ms)  # make sure we have a clock and therefore a t
    G.c = "all_specified(a, b, t)"
    G.c = "consistent_units(a, b, t)"
    with pytest.raises(ValueError):
        setattr(G, "c", "no_result_unit(a, b, t)")
    with pytest.raises(KeyError):
        setattr(G, "c", "nothing_specified(a, b, t)")
    with pytest.raises(DimensionMismatchError):
        setattr(G, "a", "all_specified(a, b, t)")
    with pytest.raises(DimensionMismatchError):
        setattr(G, "a", "all_specified(b, a, t)")
    with pytest.raises(DimensionMismatchError):
        setattr(G, "a", "consistent_units(a, b, t)")
    with pytest.raises(DimensionMismatchError):
        setattr(G, "a", "consistent_units(b, a, t)")


def test_simple_user_defined_function():
    # Make sure that it's possible to use a Python function directly, without
    # additional wrapping
    @check_units(x=1, result=1)
    def usersin(x):
        return np.sin(x)

    usersin.stateless = True

    default_dt = defaultclock.dt
    test_array = np.array([0, 1, 2, 3])
    G = NeuronGroup(
        len(test_array),
        """func = usersin(variable) : 1
                       variable : 1""",
        codeobj_class=NumpyCodeObject,
    )
    G.variable = test_array
    mon = StateMonitor(G, "func", record=True, codeobj_class=NumpyCodeObject)
    net = Network(G, mon)
    net.run(default_dt)

    assert_allclose(np.sin(test_array), mon.func_.flatten())


def test_manual_user_defined_function():
    if prefs.codegen.target != "numpy":
        pytest.skip("numpy-only test")

    default_dt = defaultclock.dt

    # User defined function without any decorators
    def foo(x, y):
        return x + y + 3 * volt

    orig_foo = foo
    # Since the function is not annotated with check units, we need to specify
    # both the units of the arguments and the return unit
    with pytest.raises(ValueError):
        Function(foo, return_unit=volt)
    with pytest.raises(ValueError):
        Function(foo, arg_units=[volt, volt])
    # If the function uses the string syntax for "same units", it needs to
    # specify the names of the arguments
    with pytest.raises(TypeError):
        Function(foo, arg_units=[volt, "x"])
    with pytest.raises(TypeError):
        Function(foo, arg_units=[volt, "x"], arg_names=["x"])  # Needs two entries

    foo = Function(foo, arg_units=[volt, volt], return_unit=volt)

    assert foo(1 * volt, 2 * volt) == 6 * volt

    # a can be any unit, b and c need to be the same unit
    def bar(a, b, c):
        return a * (b + c)

    bar = Function(
        bar,
        arg_units=[None, None, "b"],
        arg_names=["a", "b", "c"],
        return_unit=lambda a, b, c: a * b,
    )
    assert bar(2, 3 * volt, 5 * volt) == 16 * volt
    assert bar(2 * amp, 3 * volt, 5 * volt) == 16 * watt
    assert bar(2 * volt, 3 * amp, 5 * amp) == 16 * watt

    with pytest.raises(DimensionMismatchError):
        bar(2, 3 * volt, 5 * amp)

    # Incorrect argument units
    group = NeuronGroup(
        1,
        """
        dv/dt = foo(x, y)/ms : volt
        x : 1
        y : 1
        """,
    )
    net = Network(group)
    with pytest.raises(BrianObjectException) as exc:
        net.run(0 * ms, namespace={"foo": foo})
    assert exc_isinstance(exc, DimensionMismatchError)

    # Incorrect output unit
    group = NeuronGroup(
        1,
        """
        dv/dt = foo(x, y)/ms : 1
        x : volt
        y : volt
        """,
    )
    net = Network(group)
    with pytest.raises(BrianObjectException) as exc:
        net.run(0 * ms, namespace={"foo": foo})
    assert exc_isinstance(exc, DimensionMismatchError)

    G = NeuronGroup(
        1,
        """
        func = foo(x, y) : volt
        x : volt
        y : volt
        """,
    )
    G.x = 1 * volt
    G.y = 2 * volt
    mon = StateMonitor(G, "func", record=True)
    net = Network(G, mon)
    net.run(default_dt)

    assert mon[0].func == [6] * volt

    # discard units
    foo.implementations.add_numpy_implementation(orig_foo, discard_units=True)
    G = NeuronGroup(
        1,
        """
        func = foo(x, y) : volt
        x : volt
        y : volt
        """,
    )
    G.x = 1 * volt
    G.y = 2 * volt
    mon = StateMonitor(G, "func", record=True)
    net = Network(G, mon)
    net.run(default_dt)

    assert mon[0].func == [6] * volt


@pytest.mark.cpp_standalone
@pytest.mark.standalone_only
def test_manual_user_defined_function_cpp_standalone_compiler_args():
    set_device("cpp_standalone", directory=None)

    @implementation(
        "cpp",
        """
        static inline double foo(const double x, const double y)
        {
            return x + y + _THREE;
        }
        """,  # just check whether we can specify the supported compiler args,
        # only the define macro is actually used
        headers=[],
        sources=[],
        libraries=[],
        include_dirs=[],
        library_dirs=[],
        runtime_library_dirs=[],
        define_macros=[("_THREE", "3")],
    )
    @check_units(x=volt, y=volt, result=volt)
    def foo(x, y):
        return x + y + 3 * volt

    G = NeuronGroup(
        1,
        """
        func = foo(x, y) : volt
        x : volt
        y : volt
        """,
    )
    G.x = 1 * volt
    G.y = 2 * volt
    mon = StateMonitor(G, "func", record=True)
    net = Network(G, mon)
    net.run(defaultclock.dt)
    assert mon[0].func == [6] * volt


@pytest.mark.cpp_standalone
@pytest.mark.standalone_only
def test_manual_user_defined_function_cpp_standalone_wrong_compiler_args1():
    set_device("cpp_standalone", directory=None)

    @implementation(
        "cpp",
        """
        static inline double foo(const double x, const double y)
        {
            return x + y + _THREE;
        }
        """,
        some_arg=[],
    )  # non-existing argument
    @check_units(x=volt, y=volt, result=volt)
    def foo(x, y):
        return x + y + 3 * volt

    G = NeuronGroup(
        1,
        """
        func = foo(x, y) : volt
        x : volt
        y : volt
        """,
    )
    mon = StateMonitor(G, "func", record=True)
    net = Network(G, mon)
    with pytest.raises(BrianObjectException) as exc:
        net.run(defaultclock.dt, namespace={"foo": foo})
    assert exc_isinstance(exc, ValueError)


@pytest.mark.cpp_standalone
@pytest.mark.standalone_only
def test_manual_user_defined_function_cpp_standalone_wrong_compiler_args2():
    set_device("cpp_standalone", directory=None)

    @implementation(
        "cpp",
        """
        static inline double foo(const double x, const double y)
        {
            return x + y + _THREE;
        }
        """,
        headers="<stdio.h>",
    )  # existing argument, wrong value type
    @check_units(x=volt, y=volt, result=volt)
    def foo(x, y):
        return x + y + 3 * volt

    G = NeuronGroup(
        1,
        """
        func = foo(x, y) : volt
        x : volt
        y : volt
        """,
    )
    mon = StateMonitor(G, "func", record=True)
    net = Network(G, mon)
    with pytest.raises(BrianObjectException) as exc:
        net.run(defaultclock.dt, namespace={"foo": foo})
    assert exc_isinstance(exc, TypeError)


def test_manual_user_defined_function_cython_compiler_args():
    if prefs.codegen.target != "cython":
        pytest.skip("Cython-only test")

    @implementation(
        "cython",
        """
        cdef double foo(double x, const double y):
            return x + y + 3
        """,  # just check whether we can specify the supported compiler args,
        libraries=[],
        include_dirs=[],
        library_dirs=[],
        runtime_library_dirs=[],
    )
    @check_units(x=volt, y=volt, result=volt)
    def foo(x, y):
        return x + y + 3 * volt

    G = NeuronGroup(
        1,
        """
                       func = foo(x, y) : volt
                       x : volt
                       y : volt""",
    )
    G.x = 1 * volt
    G.y = 2 * volt
    mon = StateMonitor(G, "func", record=True)
    net = Network(G, mon)
    net.run(defaultclock.dt)
    assert mon[0].func == [6] * volt


def test_manual_user_defined_function_cython_wrong_compiler_args1():
    if prefs.codegen.target != "cython":
        pytest.skip("Cython-only test")

    @implementation(
        "cython",
        """
        cdef double foo(double x, const double y):
            return x + y + 3
        """,
        some_arg=[],
    )  # non-existing argument
    @check_units(x=volt, y=volt, result=volt)
    def foo(x, y):
        return x + y + 3 * volt

    G = NeuronGroup(
        1,
        """
        func = foo(x, y) : volt
        x : volt
        y : volt
        """,
    )
    mon = StateMonitor(G, "func", record=True)
    net = Network(G, mon)
    with pytest.raises(BrianObjectException) as exc:
        net.run(defaultclock.dt, namespace={"foo": foo})
    assert exc_isinstance(exc, ValueError)


def test_manual_user_defined_function_cython_wrong_compiler_args2():
    if prefs.codegen.target != "cython":
        pytest.skip("Cython-only test")

    @implementation(
        "cython",
        """
        cdef double foo(double x, const double y):
            return x + y + 3
        """,
        libraries="cstdio",
    )  # existing argument, wrong value type
    @check_units(x=volt, y=volt, result=volt)
    def foo(x, y):
        return x + y + 3 * volt

    G = NeuronGroup(
        1,
        """
        func = foo(x, y) : volt
        x : volt
        y : volt
        """,
    )
    mon = StateMonitor(G, "func", record=True)
    net = Network(G, mon)
    with pytest.raises(BrianObjectException) as exc:
        net.run(defaultclock.dt, namespace={"foo": foo})
    assert exc_isinstance(exc, TypeError)


def test_external_function_cython():
    if prefs.codegen.target != "cython":
        pytest.skip("Cython-only test")

    this_dir = os.path.abspath(os.path.dirname(__file__))

    @implementation(
        "cython",
        "from func_def_cython cimport foo",
        sources=[os.path.join(this_dir, "func_def_cython.pyx")],
    )
    @check_units(x=volt, y=volt, result=volt)
    def foo(x, y):
        return x + y + 3 * volt

    G = NeuronGroup(
        1,
        """
        func = foo(x, y) : volt
        x : volt
        y : volt
        """,
    )
    G.x = 1 * volt
    G.y = 2 * volt
    mon = StateMonitor(G, "func", record=True)
    net = Network(G, mon)
    net.run(defaultclock.dt)
    assert mon[0].func == [6] * volt


@pytest.mark.cpp_standalone
@pytest.mark.standalone_only
def test_external_function_cpp_standalone():
    set_device("cpp_standalone", directory=None)
    this_dir = os.path.abspath(os.path.dirname(__file__))
    with tempfile.TemporaryDirectory(prefix="brian_testsuite_") as tmpdir:
        # copy the test function to the temporary directory
        # this avoids issues with the file being in a directory that is not writable
        shutil.copy(os.path.join(this_dir, "func_def_cpp.h"), tmpdir)
        shutil.copy(os.path.join(this_dir, "func_def_cpp.cpp"), tmpdir)

        @implementation(
            "cpp",
            "//all code in func_def_cpp.cpp",
            headers=['"func_def_cpp.h"'],
            include_dirs=[tmpdir],
            sources=[os.path.join(tmpdir, "func_def_cpp.cpp")],
        )
        @check_units(x=volt, y=volt, result=volt)
        def foo(x, y):
            return x + y + 3 * volt

        G = NeuronGroup(
            1,
            """
            func = foo(x, y) : volt
            x : volt
            y : volt
            """,
        )
        G.x = 1 * volt
        G.y = 2 * volt
        mon = StateMonitor(G, "func", record=True)
        net = Network(G, mon)
        net.run(defaultclock.dt)
        assert mon[0].func == [6] * volt


@pytest.mark.codegen_independent
def test_user_defined_function_discarding_units():
    # A function with units that should discard units also inside the function
    @implementation("numpy", discard_units=True)
    @check_units(v=volt, result=volt)
    def foo(v):
        return v + 3 * volt  # this normally raises an error for unitless v

    assert foo(5 * volt) == 8 * volt

    # Test the function that is used during a run
    assert foo.implementations[NumpyCodeObject].get_code(None)(5) == 8


@pytest.mark.codegen_independent
def test_user_defined_function_discarding_units_2():
    # Add a numpy implementation explicitly (as in TimedArray)
    unit = volt

    @check_units(v=volt, result=unit)
    def foo(v):
        return v + 3 * unit  # this normally raises an error for unitless v

    foo = Function(pyfunc=foo)

    def unitless_foo(v):
        return v + 3

    foo.implementations.add_implementation("numpy", code=unitless_foo)

    assert foo(5 * volt) == 8 * volt

    # Test the function that is used during a run
    assert foo.implementations[NumpyCodeObject].get_code(None)(5) == 8


@pytest.mark.codegen_independent
def test_function_implementation_container():
    import brian2.codegen.targets as targets

    class ACodeGenerator(CodeGenerator):
        class_name = "A Language"

    class BCodeGenerator(CodeGenerator):
        class_name = "B Language"

    class ACodeObject(CodeObject):
        generator_class = ACodeGenerator
        class_name = "A"

    class A2CodeObject(CodeObject):
        generator_class = ACodeGenerator
        class_name = "A2"

    class BCodeObject(CodeObject):
        generator_class = BCodeGenerator
        class_name = "B"

    # Register the code generation targets
    _previous_codegen_targets = set(targets.codegen_targets)
    targets.codegen_targets = {ACodeObject, BCodeObject}

    @check_units(x=volt, result=volt)
    def foo(x):
        return x

    f = Function(foo)

    container = f.implementations

    # inserting into the container with a CodeGenerator class
    container.add_implementation(BCodeGenerator, code="implementation B language")
    assert container[BCodeGenerator].get_code(None) == "implementation B language"

    # inserting into the container with a CodeObject class
    container.add_implementation(ACodeObject, code="implementation A CodeObject")
    assert container[ACodeObject].get_code(None) == "implementation A CodeObject"

    # inserting into the container with a name of a CodeGenerator
    container.add_implementation("A Language", "implementation A Language")
    assert container["A Language"].get_code(None) == "implementation A Language"
    assert container[ACodeGenerator].get_code(None) == "implementation A Language"
    assert container[A2CodeObject].get_code(None) == "implementation A Language"

    # inserting into the container with a name of a CodeObject
    container.add_implementation("B", "implementation B CodeObject")
    assert container["B"].get_code(None) == "implementation B CodeObject"
    assert container[BCodeObject].get_code(None) == "implementation B CodeObject"

    with pytest.raises(KeyError):
        container["unknown"]

    # some basic dictionary properties
    assert len(container) == 4
    assert {key for key in container} == {
        "A Language",
        "B",
        ACodeObject,
        BCodeGenerator,
    }

    # Restore the previous codegeneration targets
    targets.codegen_targets = _previous_codegen_targets


def test_function_dependencies_cython():
    if prefs.codegen.target != "cython":
        pytest.skip("cython-only test")

    @implementation(
        "cython",
        """
        cdef float foo(float x):
            return 42*0.001
        """,
    )
    @check_units(x=volt, result=volt)
    def foo(x):
        return 42 * mV

    # Second function with an independent implementation for numpy and an
    # implementation for C++ that makes use of the previous function.

    @implementation(
        "cython",
        """
        cdef float bar(float x):
            return 2*foo(x)
        """,
        dependencies={"foo": foo},
    )
    @check_units(x=volt, result=volt)
    def bar(x):
        return 84 * mV

    G = NeuronGroup(5, "v : volt")
    G.run_regularly("v = bar(v)")
    net = Network(G)
    net.run(defaultclock.dt)

    assert_allclose(G.v_[:], 84 * 0.001)


def test_function_dependencies_cython_rename():
    if prefs.codegen.target != "cython":
        pytest.skip("cython-only test")

    @implementation(
        "cython",
        """
        cdef float _foo(float x):
            return 42*0.001
        """,
        name="_foo",
    )
    @check_units(x=volt, result=volt)
    def foo(x):
        return 42 * mV

    # Second function with an independent implementation for numpy and an
    # implementation for C++ that makes use of the previous function.

    @implementation(
        "cython",
        """
        cdef float bar(float x):
            return 2*my_foo(x)
        """,
        dependencies={"my_foo": foo},
    )
    @check_units(x=volt, result=volt)
    def bar(x):
        return 84 * mV

    G = NeuronGroup(5, "v : volt")
    G.run_regularly("v = bar(v)")
    net = Network(G)
    net.run(defaultclock.dt)

    assert_allclose(G.v_[:], 84 * 0.001)


def test_function_dependencies_numpy():
    if prefs.codegen.target != "numpy":
        pytest.skip("numpy-only test")

    @implementation(
        "cpp",
        """
    float foo(float x)
    {
        return 42*0.001;
    }""",
    )
    @check_units(x=volt, result=volt)
    def foo(x):
        return 42 * mV

    # Second function with an independent implementation for C++ and an
    # implementation for numpy that makes use of the previous function.

    # Note that we don't need to use the explicit dependencies mechanism for
    # numpy, since the Python function stores a reference to the referenced
    # function directly

    @implementation(
        "cpp",
        """
        float bar(float x)
        {
            return 84*0.001;
        }""",
    )
    @check_units(x=volt, result=volt)
    def bar(x):
        return 2 * foo(x)

    G = NeuronGroup(5, "v : volt")
    G.run_regularly("v = bar(v)")
    net = Network(G)
    net.run(defaultclock.dt)

    assert_allclose(G.v_[:], 84 * 0.001)


@pytest.mark.standalone_compatible
def test_repeated_function_dependencies():
    # each of the binomial functions adds randn as a depency, see #988
    test_neuron = NeuronGroup(
        1,
        "x : 1",
        namespace={
            "bino_1": BinomialFunction(10, 0.5),
            "bino_2": BinomialFunction(10, 0.6),
        },
    )
    test_neuron.x = "bino_1()+bino_2()"

    run(0 * ms)


@pytest.mark.standalone_compatible
def test_binomial():
    binomial_f_approximated = BinomialFunction(100, 0.1, approximate=True)
    binomial_f = BinomialFunction(100, 0.1, approximate=False)

    # Just check that it does not raise an error and that it produces some
    # values
    G = NeuronGroup(
        1,
        """
        x : 1
        y : 1
        """,
    )
    G.run_regularly(
        """
        x = binomial_f_approximated()
        y = binomial_f()
        """
    )
    mon = StateMonitor(G, ["x", "y"], record=0)
    run(1 * ms)
    assert np.var(mon[0].x) > 0
    assert np.var(mon[0].y) > 0


@pytest.mark.standalone_compatible
def test_poisson():
    # Just check that it does not raise an error and that it produces some
    # values
    G = NeuronGroup(
        5,
        """
        l : 1
        x : integer
        y : integer
        z : integer
        """,
    )
    G.l = [0, 1, 5, 15, 25]
    G.run_regularly(
        """
        x = poisson(l)
        y = poisson(5)
        z = poisson(0)
        """
    )
    mon = StateMonitor(G, ["x", "y", "z"], record=True)
    run(100 * defaultclock.dt)
    assert_equal(mon.x[0], 0)
    assert all(np.var(mon.x[1:], axis=1) > 0)
    assert all(np.var(mon.y, axis=1) > 0)
    assert_equal(mon.z, 0)


def test_declare_types():
    if prefs.codegen.target != "numpy":
        pytest.skip("numpy-only test")

    @declare_types(a="integer", b="float", result="highest")
    def f(a, b):
        return a * b

    assert f._arg_types == ["integer", "float"]
    assert f._return_type == "highest"

    @declare_types(b="float")
    def f(a, b, c):
        return a * b * c

    assert f._arg_types == ["any", "float", "any"]
    assert f._return_type == "float"

    def bad_argtype():
        @declare_types(b="floating")
        def f(a, b, c):
            return a * b * c

    with pytest.raises(ValueError):
        bad_argtype()

    def bad_argname():
        @declare_types(d="floating")
        def f(a, b, c):
            return a * b * c

    with pytest.raises(ValueError):
        bad_argname()

    @check_units(a=volt, b=1)
    @declare_types(a="float", b="integer")
    def f(a, b):
        return a * b

    @declare_types(a="float", b="integer")
    @check_units(a=volt, b=1)
    def f(a, b):
        return a * b

    def bad_units():
        @declare_types(a="integer", b="float")
        @check_units(a=volt, b=1, result=volt)
        def f(a, b):
            return a * b

        eqs = """
        dv/dt = f(v, 1)/second : 1
        """
        G = NeuronGroup(1, eqs)
        Network(G).run(1 * ms)

    with pytest.raises(BrianObjectException) as exc:
        bad_units()
    assert exc_isinstance(exc, TypeError)

    def bad_type():
        @implementation("numpy", discard_units=True)
        @declare_types(a="float", result="float")
        @check_units(a=1, result=1)
        def f(a):
            return a

        eqs = """
        a : integer
        dv/dt = f(a)*v/second : 1
        """
        G = NeuronGroup(1, eqs)
        Network(G).run(1 * ms)

    with pytest.raises(BrianObjectException) as exc:
        bad_type()
    assert exc_isinstance(exc, TypeError)


def test_multiple_stateless_function_calls():
    # Check that expressions such as rand() + rand() (which might be incorrectly
    # simplified to 2*rand()) raise an error
    G = NeuronGroup(1, "dv/dt = (rand() - rand())/second : 1")
    net = Network(G)
    with pytest.raises(BrianObjectException) as exc:
        net.run(0 * ms)
    assert exc_isinstance(exc, NotImplementedError)
    G2 = NeuronGroup(1, "v:1", threshold="v>1", reset="v=rand() - rand()")
    net2 = Network(G2)
    with pytest.raises(BrianObjectException) as exc:
        net2.run(0 * ms)
    assert exc_isinstance(exc, NotImplementedError)
    G3 = NeuronGroup(1, "v:1")
    G3.run_regularly("v = rand() - rand()")
    net3 = Network(G3)
    with pytest.raises(BrianObjectException) as exc:
        net3.run(0 * ms)
    assert exc_isinstance(exc, NotImplementedError)
    G4 = NeuronGroup(1, "x : 1")
    # Verify that synaptic equations are checked as well, see #1146
    S = Synapses(G4, G4, "dy/dt = (rand() - rand())/second : 1 (clock-driven)")
    S.connect()
    net = Network(G4, S)
    with pytest.raises(BrianObjectException) as exc:
        net.run(0 * ms)
    assert exc_isinstance(exc, NotImplementedError)


@pytest.mark.codegen_independent
def test_parse_dimension_errors():
    from brian2.parsing.expressions import parse_expression_dimensions

    @check_units(x=1, result=1)
    def foo(x):
        return x

    # Function call with keyword arguments
    with pytest.raises(ValueError):
        parse_expression_dimensions("foo(a=1, b=2)", {"foo": foo})
    # Unknown function
    with pytest.raises(SyntaxError):
        parse_expression_dimensions("bar(1, 2)", {"foo": foo})
    # Function without unit definition
    with pytest.raises(ValueError):
        parse_expression_dimensions("bar(1, 2)", {"bar": lambda x, y: x + y})
    # Function with wrong number of arguments
    with pytest.raises(SyntaxError):
        parse_expression_dimensions("foo(1, 2)", {"foo": foo})


if __name__ == "__main__":
    # prefs.codegen.target = 'numpy'
    import time

    from _pytest.outcomes import Skipped

    from brian2 import prefs

    for f in [
        test_constants_sympy,
        test_constants_values,
        test_math_functions,
        test_clip,
        test_bool_to_int,
        test_timestep_function,
        test_timestep_function_during_run,
        test_user_defined_function,
        test_user_defined_function_units,
        test_simple_user_defined_function,
        test_manual_user_defined_function,
        test_external_function_cython,
        test_user_defined_function_discarding_units,
        test_user_defined_function_discarding_units_2,
        test_function_implementation_container,
        test_function_dependencies_numpy,
        test_function_dependencies_cython,
        test_function_dependencies_cython_rename,
        test_repeated_function_dependencies,
        test_binomial,
        test_poisson,
        test_declare_types,
        test_multiple_stateless_function_calls,
    ]:
        try:
            start = time.time()
            f()
            print("Test", f.__name__, "took", time.time() - start)
        except Skipped as e:
            print("Skipping test", f.__name__, e)
