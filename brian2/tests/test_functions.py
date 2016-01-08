from nose import SkipTest, with_setup
from nose.plugins.attrib import attr
from numpy.testing import assert_equal, assert_raises, assert_allclose

from brian2 import *
from brian2.parsing.sympytools import str_to_sympy, sympy_to_str
from brian2.utils.logger import catch_logs
from brian2.devices.device import reinit_devices

@attr('codegen-independent')
def test_constants_sympy():
    '''
    Make sure that symbolic constants are understood correctly by sympy
    '''
    assert sympy_to_str(str_to_sympy('1.0/inf')) == '0'
    assert sympy_to_str(str_to_sympy('sin(pi)')) == '0'
    assert sympy_to_str(str_to_sympy('log(e)')) == '1'


def test_constants_values():
    '''
    Make sure that symbolic constants use the correct values in code
    '''
    G = NeuronGroup(1, 'v : 1')
    G.v = 'pi'
    assert G.v == np.pi
    G.v = 'e'
    assert G.v == np.e
    G.v = 'inf'
    assert G.v == np.inf


def test_math_functions():
    '''
    Test that math functions give the same result, regardless of whether used
    directly or in generated Python or C++ code.
    '''
    default_dt = defaultclock.dt
    test_array = np.array([-1, -0.5, 0, 0.5, 1])
    def int_(x):
        return array(x, dtype=int)
    int_.__name__ = 'int'

    with catch_logs() as _:  # Let's suppress warnings about illegal values
        # Functions with a single argument
        for func in [cos, tan, sinh, cosh, tanh,
                     arcsin, arccos, arctan,
                     log, log10,
                     exp, np.sqrt,
                     np.ceil, np.floor, np.sign, int_]:

            # Calculate the result directly
            numpy_result = func(test_array)

            # Calculate the result in a somewhat complicated way by using a
            # subexpression in a NeuronGroup
            if func.__name__ == 'absolute':
                # we want to use the name abs instead of absolute
                func_name = 'abs'
            else:
                func_name = func.__name__
            G = NeuronGroup(len(test_array),
                            '''func = {func}(variable) : 1
                               variable : 1'''.format(func=func_name))
            G.variable = test_array
            mon = StateMonitor(G, 'func', record=True)
            net = Network(G, mon)
            net.run(default_dt)

            assert_allclose(numpy_result, mon.func_.flatten(),
                            err_msg='Function %s did not return the correct values' % func.__name__)

        # Functions/operators
        scalar = 3
        for func, operator in [(np.power, '**'), (np.mod, '%')]:

            # Calculate the result directly
            numpy_result = func(test_array, scalar)

            # Calculate the result in a somewhat complicated way by using a
            # subexpression in a NeuronGroup
            G = NeuronGroup(len(test_array),
                            '''func = variable {op} scalar : 1
                               variable : 1'''.format(op=operator))
            G.variable = test_array
            mon = StateMonitor(G, 'func', record=True)
            net = Network(G, mon)
            net.run(default_dt)

            assert_allclose(numpy_result, mon.func_.flatten(),
                            err_msg='Function %s did not return the correct values' % func.__name__)

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_bool_to_int():
    # Test that boolean expressions and variables are correctly converted into
    # integers
    G = NeuronGroup(2, '''
                       intexpr1 = int(bool_var) : integer
                       intexpr2 = int(float_var > 1.0) : integer
                       bool_var : boolean
                       float_var : 1
                       ''')
    G.bool_var = [True, False]
    G.float_var = [2.0, 0.5]
    s_mon = StateMonitor(G, ['intexpr1', 'intexpr2'], record=True)
    run(defaultclock.dt)
    assert_equal(s_mon.intexpr1.flatten(), [1, 0])
    assert_equal(s_mon.intexpr2.flatten(), [1, 0])

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_user_defined_function():
    @implementation('cpp',"""
                inline double usersin(double x)
                {
                    return sin(x);
                }
                """)
    @implementation('cython', '''
            cdef double usersin(double x):
                return sin(x)
            ''')
    @check_units(x=1, result=1)
    def usersin(x):
        return np.sin(x)

    default_dt = defaultclock.dt
    test_array = np.array([0, 1, 2, 3])
    G = NeuronGroup(len(test_array),
                    '''func = usersin(variable) : 1
                              variable : 1''')
    G.variable = test_array
    mon = StateMonitor(G, 'func', record=True)
    run(default_dt)
    assert_equal(np.sin(test_array), mon.func_.flatten())


@with_setup(teardown=reinit_devices)
def test_user_defined_function_units():
    '''
    Test the preparation of functions for use in code with check_units.
    '''
    if prefs.codegen.target != 'numpy':
        raise SkipTest('numpy-only test')

    def nothing_specified(x, y, z):
        return x*(y+z)

    no_result_unit = check_units(x=1, y=second, z=second)(nothing_specified)
    one_arg_missing = check_units(x=1, z=second, result=second)(nothing_specified)
    all_specified = check_units(x=1, y=second, z=second, result=second)(nothing_specified)

    G = NeuronGroup(1, '''a : 1
                          b : second
                          c : second''',
                    namespace={'nothing_specified': nothing_specified,
                               'no_result_unit': no_result_unit,
                               'one_arg_missing': one_arg_missing,
                               'all_specified': all_specified})
    net = Network(G)
    net.run(0*ms)  # make sure we have a clock and therefore a t
    G.c = 'all_specified(a, b, t)'
    assert_raises(ValueError,
                  lambda: setattr(G, 'c', 'one_arg_missing(a, b, t)'))
    assert_raises(ValueError,
                  lambda: setattr(G, 'c', 'no_result_unit(a, b, t)'))
    assert_raises(KeyError,
                  lambda: setattr(G, 'c', 'nothing_specified(a, b, t)'))
    assert_raises(DimensionMismatchError,
                  lambda: setattr(G, 'a', 'all_specified(a, b, t)'))
    assert_raises(DimensionMismatchError,
                  lambda: setattr(G, 'a', 'all_specified(b, a, t)'))


def test_simple_user_defined_function():
    # Make sure that it's possible to use a Python function directly, without
    # additional wrapping
    @check_units(x=1, result=1)
    def usersin(x):
        return np.sin(x)

    default_dt = defaultclock.dt
    test_array = np.array([0, 1, 2, 3])
    G = NeuronGroup(len(test_array),
                    '''func = usersin(variable) : 1
                              variable : 1''',
                    codeobj_class=NumpyCodeObject)
    G.variable = test_array
    mon = StateMonitor(G, 'func', record=True, codeobj_class=NumpyCodeObject)
    net = Network(G, mon)
    net.run(default_dt)

    assert_equal(np.sin(test_array), mon.func_.flatten())

    # Check that it raises an error for C++
    try:
        import scipy.weave
        G = NeuronGroup(len(test_array),
                        '''func = usersin(variable) : 1
                              variable : 1''',
                        codeobj_class=WeaveCodeObject)
        mon = StateMonitor(G, 'func', record=True,
                           codeobj_class=WeaveCodeObject)
        net = Network(G, mon)
        # This looks a bit odd -- we have to get usersin into the namespace of
        # the lambda expression
        assert_raises(NotImplementedError,
                      lambda usersin: net.run(0.1*ms), usersin)
    except ImportError:
        pass


def test_manual_user_defined_function():
    if prefs.codegen.target != 'numpy':
        raise SkipTest('numpy-only test')

    default_dt = defaultclock.dt

    # User defined function without any decorators
    def foo(x, y):
        return x + y + 3*volt
    orig_foo = foo
    # Since the function is not annotated with check units, we need to specify
    # both the units of the arguments and the return unit
    assert_raises(ValueError, lambda: Function(foo, return_unit=volt))
    assert_raises(ValueError, lambda: Function(foo, arg_units=[volt, volt]))
    foo = Function(foo, arg_units=[volt, volt], return_unit=volt)

    assert foo(1*volt, 2*volt) == 6*volt

    # Incorrect argument units
    group = NeuronGroup(1, '''
                       dv/dt = foo(x, y)/ms : volt
                       x : 1
                       y : 1''')
    net = Network(group)
    assert_raises(DimensionMismatchError,
                  lambda: net.run(0*ms, namespace={ 'foo': foo}))

    # Incorrect output unit
    group = NeuronGroup(1, '''
                       dv/dt = foo(x, y)/ms : 1
                       x : volt
                       y : volt''')
    net = Network(group)
    assert_raises(DimensionMismatchError,
                  lambda: net.run(0*ms, namespace={'foo': foo}))

    G = NeuronGroup(1, '''
                       func = foo(x, y) : volt
                       x : volt
                       y : volt''')
    G.x = 1*volt
    G.y = 2*volt
    mon = StateMonitor(G, 'func', record=True)
    net = Network(G, mon)
    net.run(default_dt)

    assert mon[0].func == [6] * volt

    # discard units
    foo.implementations.add_numpy_implementation(orig_foo,
                                                 discard_units=True)
    G = NeuronGroup(1, '''
                       func = foo(x, y) : volt
                       x : volt
                       y : volt''')
    G.x = 1*volt
    G.y = 2*volt
    mon = StateMonitor(G, 'func', record=True)
    net = Network(G, mon)
    net.run(default_dt)

    assert mon[0].func == [6] * volt


def test_manual_user_defined_function_weave():
    if prefs.codegen.target != 'weave':
        raise SkipTest('weave-only test')

    # User defined function without any decorators
    def foo(x, y):
        return x + y + 3*volt

    foo = Function(foo, arg_units=[volt, volt], return_unit=volt)

    code = {'support_code': '''
    inline double foo(const double x, const double y)
    {
        return x + y + 3;
    }
    '''}

    foo.implementations.add_implementation('cpp', code)

    G = NeuronGroup(1, '''
                       func = foo(x, y) : volt
                       x : volt
                       y : volt''')
    G.x = 1*volt
    G.y = 2*volt
    mon = StateMonitor(G, 'func', record=True)
    net = Network(G, mon)
    net.run(defaultclock.dt)
    assert mon[0].func == [6] * volt


@attr('codegen-independent')
def test_user_defined_function_discarding_units():
    # A function with units that should discard units also inside the function
    @implementation('numpy', discard_units=True)
    @check_units(v=volt, result=volt)
    def foo(v):
        return v + 3*volt  # this normally raises an error for unitless v

    assert foo(5*volt) == 8*volt

    # Test the function that is used during a run
    assert foo.implementations[NumpyCodeObject].get_code(None)(5) == 8


@attr('codegen-independent')
def test_user_defined_function_discarding_units_2():
    # Add a numpy implementation explicitly (as in TimedArray)
    unit = volt
    @check_units(v=volt, result=unit)
    def foo(v):
        return v + 3*unit  # this normally raises an error for unitless v

    foo = Function(pyfunc=foo)
    def unitless_foo(v):
        return v + 3

    foo.implementations.add_implementation('numpy', code=unitless_foo)

    assert foo(5*volt) == 8*volt

    # Test the function that is used during a run
    assert foo.implementations[NumpyCodeObject].get_code(None)(5) == 8


@attr('codegen-independent')
def test_function_implementation_container():
    import brian2.codegen.targets as targets

    class ACodeGenerator(CodeGenerator):
        class_name = 'A Language'

    class BCodeGenerator(CodeGenerator):
        class_name = 'B Language'

    class ACodeObject(CodeObject):
        generator_class = ACodeGenerator
        class_name = 'A'

    class A2CodeObject(CodeObject):
        generator_class = ACodeGenerator
        class_name = 'A2'

    class BCodeObject(CodeObject):
        generator_class = BCodeGenerator
        class_name = 'B'


    # Register the code generation targets
    _previous_codegen_targets = set(targets.codegen_targets)
    targets.codegen_targets = {ACodeObject, BCodeObject}

    @check_units(x=volt, result=volt)
    def foo(x):
        return x
    f = Function(foo)

    container = f.implementations

    # inserting into the container with a CodeGenerator class
    container.add_implementation(BCodeGenerator, code='implementation B language')
    assert container[BCodeGenerator].get_code(None) == 'implementation B language'

    # inserting into the container with a CodeObject class
    container.add_implementation(ACodeObject, code='implementation A CodeObject')
    assert container[ACodeObject].get_code(None) == 'implementation A CodeObject'

    # inserting into the container with a name of a CodeGenerator
    container.add_implementation('A Language', 'implementation A Language')
    assert container['A Language'].get_code(None) == 'implementation A Language'
    assert container[ACodeGenerator].get_code(None) == 'implementation A Language'
    assert container[A2CodeObject].get_code(None) == 'implementation A Language'

    # inserting into the container with a name of a CodeObject
    container.add_implementation('B', 'implementation B CodeObject')
    assert container['B'].get_code(None) == 'implementation B CodeObject'
    assert container[BCodeObject].get_code(None) == 'implementation B CodeObject'

    assert_raises(KeyError, lambda: container['unknown'])

    # some basic dictionary properties
    assert len(container) == 4
    assert set((key for key in container)) == {'A Language', 'B', ACodeObject,
                                               BCodeGenerator}

    # Restore the previous codegeneration targets
    targets.codegen_targets = _previous_codegen_targets


def test_function_dependencies_weave():
    if prefs.codegen.target != 'weave':
        raise SkipTest('weave-only test')

    @implementation('cpp', '''
    float foo(float x)
    {
        return 42*0.001;
    }''')
    @check_units(x=volt, result=volt)
    def foo(x):
        return 42*mV

    # Second function with an independent implementation for numpy and an
    # implementation for C++ that makes use of the previous function.

    @implementation('cpp', '''
    float bar(float x)
    {
        return 2*foo(x);
    }''', dependencies={'foo': foo})
    @check_units(x=volt, result=volt)
    def bar(x):
        return 84*mV

    G = NeuronGroup(5, 'v : volt')
    G.run_regularly('v = bar(v)')
    net = Network(G)
    net.run(defaultclock.dt)

    assert_allclose(G.v_[:], 84*0.001)


def test_function_dependencies_cython():
    if prefs.codegen.target != 'cython':
        raise SkipTest('cython-only test')

    @implementation('cython', '''
    cdef float foo(float x):
        return 42*0.001
    ''')
    @check_units(x=volt, result=volt)
    def foo(x):
        return 42*mV

    # Second function with an independent implementation for numpy and an
    # implementation for C++ that makes use of the previous function.

    @implementation('cython', '''
    cdef float bar(float x):
        return 2*foo(x)
    ''', dependencies={'foo': foo})
    @check_units(x=volt, result=volt)
    def bar(x):
        return 84*mV

    G = NeuronGroup(5, 'v : volt')
    G.run_regularly('v = bar(v)')
    net = Network(G)
    net.run(defaultclock.dt)

    assert_allclose(G.v_[:], 84*0.001)


def test_function_dependencies_numpy():
    if prefs.codegen.target != 'numpy':
        raise SkipTest('numpy-only test')

    @implementation('cpp', '''
    float foo(float x)
    {
        return 42*0.001;
    }''')
    @check_units(x=volt, result=volt)
    def foo(x):
        return 42*mV

    # Second function with an independent implementation for C++ and an
    # implementation for numpy that makes use of the previous function.

    # Note that we don't need to use the explicit dependencies mechanism for
    # numpy, since the Python function stores a reference to the referenced
    # function directly

    @implementation('cpp', '''
    float bar(float x)
    {
        return 84*0.001;
    }''')
    @check_units(x=volt, result=volt)
    def bar(x):
        return 2*foo(x)

    G = NeuronGroup(5, 'v : volt')
    G.run_regularly('v = bar(v)')
    net = Network(G)
    net.run(defaultclock.dt)

    assert_allclose(G.v_[:], 84*0.001)

@attr('standalone-compatible')
@with_setup(teardown=reinit_devices)
def test_binomial():
    binomial_f_approximated = BinomialFunction(100, 0.1, approximate=True)
    binomial_f = BinomialFunction(100, 0.1, approximate=False)

    # Just check that it does not raise an error and that it produces some
    # values
    G = NeuronGroup(1, '''x : 1
                          y : 1''')
    G.run_regularly('''x = binomial_f_approximated()
                       y = binomial_f()''')
    mon = StateMonitor(G, ['x', 'y'], record=0)
    run(1*ms)
    assert np.var(mon[0].x) > 0
    assert np.var(mon[0].y) > 0


def test_declare_types():
    if prefs.codegen.target != 'numpy':
        raise SkipTest('numpy-only test')

    @declare_types(a='integer', b='float', result='highest')
    def f(a, b):
        return a*b
    assert f._arg_types==['integer', 'float']
    assert f._return_type == 'highest'

    @declare_types(b='float')
    def f(a, b, c):
        return a*b*c
    assert f._arg_types==['any', 'float', 'any']
    assert f._return_type == 'float'

    def bad_argtype():
        @declare_types(b='floating')
        def f(a, b, c):
            return a*b*c
    assert_raises(ValueError, bad_argtype)

    def bad_argname():
        @declare_types(d='floating')
        def f(a, b, c):
            return a*b*c
    assert_raises(ValueError, bad_argname)

    @check_units(a=volt, b=1)
    @declare_types(a='float', b='integer')
    def f(a, b):
        return a*b

    @declare_types(a='float', b='integer')
    @check_units(a=volt, b=1)
    def f(a, b):
        return a*b

    def bad_units():
        @declare_types(a='integer', b='float')
        @check_units(a=volt, b=1, result=volt)
        def f(a, b):
            return a*b
        eqs = '''
        dv/dt = f(v, 1)/second : 1
        '''
        G = NeuronGroup(1, eqs)
        Network(G).run(1*ms)
    assert_raises(TypeError, bad_units)

    def bad_type():
        @implementation('numpy', discard_units=True)
        @declare_types(a='float', result='float')
        @check_units(a=1, result=1)
        def f(a):
            return a
        eqs = '''
        a : integer
        dv/dt = f(a)*v/second : 1
        '''
        G = NeuronGroup(1, eqs)
        Network(G).run(1*ms)
    assert_raises(TypeError, bad_type)


def test_multiple_stateless_function_calls():
    # Check that expressions such as rand() + rand() (which might be incorrectly
    # simplified to 2*rand()) raise an error
    G = NeuronGroup(1, 'dv/dt = (rand() - rand())/second : 1')
    net = Network(G)
    assert_raises(NotImplementedError, lambda: net.run(0*ms))
    G2 = NeuronGroup(1, 'v:1', threshold='v>1', reset='v=rand() - rand()')
    net2 = Network(G2)
    assert_raises(NotImplementedError, lambda: net2.run(0*ms))
    G3 = NeuronGroup(1, 'v:1')
    G3.run_regularly('v = rand() - rand()')
    net3 = Network(G3)
    assert_raises(NotImplementedError, lambda: net3.run(0*ms))

if __name__ == '__main__':
    from brian2 import prefs
    # prefs.codegen.target = 'numpy'
    import time
    for f in [
            test_constants_sympy,
            test_constants_values,
            test_math_functions,
            test_bool_to_int,
            test_user_defined_function,
            test_user_defined_function_units,
            test_simple_user_defined_function,
            test_manual_user_defined_function,
            test_manual_user_defined_function_weave,
            test_user_defined_function_discarding_units,
            test_user_defined_function_discarding_units_2,
            test_function_implementation_container,
            test_function_dependencies_numpy,
            test_function_dependencies_weave,
            test_function_dependencies_cython,
            test_binomial,
            test_declare_types,
            test_multiple_stateless_function_calls,
            ]:
        try:
            start = time.time()
            f()
            print 'Test', f.__name__, 'took', time.time()-start
        except SkipTest as e:
            print 'Skipping test', f.__name__, e
