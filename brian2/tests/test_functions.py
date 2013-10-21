from nose.plugins.skip import SkipTest
import numpy as np
from numpy.testing import assert_equal, assert_raises

from brian2 import *
from brian2.utils.logger import catch_logs

# We can only test C++ if weave is availabe
try:
    import scipy.weave
    codeobj_classes = [WeaveCodeObject, NumpyCodeObject]
except ImportError:
    # Can't test C++
    codeobj_classes = [NumpyCodeObject]


def test_math_functions():
    '''
    Test that math functions give the same result, regardless of whether used
    directly or in generated Python or C++ code.
    '''
    test_array = np.array([-1, -0.5, 0, 0.5, 1])

    with catch_logs() as _:  # Let's suppress warnings about illegal values        
        for codeobj_class in codeobj_classes:
            
            # Functions with a single argument
            for func in [sin, cos, tan, sinh, cosh, tanh,
                         arcsin, arccos, arctan,
                         exp, log, log10,
                         np.sqrt, np.ceil, np.floor, np.abs]:
                
                # Calculate the result directly
                numpy_result = func(test_array)
                
                # Calculate the result in a somewhat complicated way by using a
                # static equation in a NeuronGroup
                clock = Clock()
                if func.__name__ == 'absolute':
                    # we want to use the name abs instead of absolute
                    func_name = 'abs'
                else:
                    func_name = func.__name__
                G = NeuronGroup(len(test_array),
                                '''func = {func}(variable) : 1
                                   variable : 1'''.format(func=func_name),
                                   clock=clock,
                                   codeobj_class=codeobj_class)
                G.variable = test_array
                mon = StateMonitor(G, 'func', record=True)
                net = Network(G, mon)
                net.run(clock.dt)
                
                assert_equal(numpy_result, mon.func_.flatten(),
                             'Function %s did not return the correct values' % func.__name__)
            
            # Functions/operators
            scalar = 3
            # TODO: We are not testing the modulo operator here since it does
            #       not work for double values in C
            for func, operator in [(np.power, '**')]:
                
                # Calculate the result directly
                numpy_result = func(test_array, scalar)
                
                # Calculate the result in a somewhat complicated way by using a
                # static equation in a NeuronGroup
                clock = Clock()
                G = NeuronGroup(len(test_array),
                                '''func = variable {op} scalar : 1
                                   variable : 1'''.format(op=operator),
                                   clock=clock,
                                   codeobj_class=codeobj_class)
                G.variable = test_array
                mon = StateMonitor(G, 'func', record=True)
                net = Network(G, mon)
                net.run(clock.dt)
                
                assert_equal(numpy_result, mon.func_.flatten(),
                             'Function %s did not return the correct values' % func.__name__)


def test_user_defined_function():
    @make_function(codes={
        'cpp':{
            'support_code':"""
                inline double usersin(double x)
                {
                    return sin(x);
                }
                """,
            'hashdefine_code':'',
            },
        })
    @check_units(x=1, result=1)
    def usersin(x):
        return np.sin(x)

    test_array = np.array([0, 1, 2, 3])
    for codeobj_class in codeobj_classes:
        G = NeuronGroup(len(test_array),
                        '''func = usersin(variable) : 1
                                  variable : 1''',
                        codeobj_class=codeobj_class)
        G.variable = test_array
        mon = StateMonitor(G, 'func', record=True)
        net = Network(G, mon)
        net.run(defaultclock.dt)

        assert_equal(np.sin(test_array), mon.func_.flatten())


def test_simple_user_defined_function():
    # Make sure that it's possible to use a Python function directly, without
    # additional wrapping
    @check_units(x=1, result=1)
    def usersin(x):
        return np.sin(x)

    test_array = np.array([0, 1, 2, 3])
    G = NeuronGroup(len(test_array),
                    '''func = usersin(variable) : 1
                              variable : 1''',
                    codeobj_class=NumpyCodeObject)
    G.variable = test_array
    mon = StateMonitor(G, 'func', record=True)
    net = Network(G, mon)
    net.run(defaultclock.dt)

    assert_equal(np.sin(test_array), mon.func_.flatten())

    # Check that it raises an error for C++
    if WeaveCodeObject in codeobj_classes:
        G = NeuronGroup(len(test_array),
                        '''func = usersin(variable) : 1
                              variable : 1''',
                        codeobj_class=WeaveCodeObject)
        mon = StateMonitor(G, 'func', record=True)
        net = Network(G, mon)
        # This looks a bit odd -- we have to get usersin into the namespace of
        # the lambda expression
        assert_raises(NotImplementedError,
                      lambda usersin: net.run(0.1*ms), usersin)


def test_manual_user_defined_function():
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
    assert_raises(DimensionMismatchError, lambda: NeuronGroup(1, '''
                       dv/dt = foo(x, y)/ms : volt
                       x : 1
                       y : 1''', namespace={'foo': foo}))

    # Incorrect output unit
    assert_raises(DimensionMismatchError, lambda: NeuronGroup(1, '''
                       dv/dt = foo(x, y)/ms : 1
                       x : volt
                       y : volt''', namespace={'foo': foo}))

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

    # discard units
    from brian2.codegen.functions import add_numpy_implementation
    add_numpy_implementation(foo, orig_foo, discard_units=True)
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

    # Test C++ implementation
    if WeaveCodeObject in codeobj_classes:
        code = {'support_code': '''
        inline double foo(const double x, const double y)
        {
            return x + y + 3;
        }
        '''}
        from brian2.codegen.functions import add_implementations
        add_implementations(foo, codes={'cpp': code})

        G = NeuronGroup(1, '''
                           func = foo(x, y) : volt
                           x : volt
                           y : volt''',
                        codeobj_class=WeaveCodeObject)
        G.x = 1*volt
        G.y = 2*volt
        mon = StateMonitor(G, 'func', record=True)
        net = Network(G, mon)
        net.run(defaultclock.dt)
        assert mon[0].func == [6] * volt


def test_add_implementations():
    if not WeaveCodeObject in codeobj_classes:
        raise SkipTest('No weave support')

    def foo(x):
        return x
    foo = Function(foo, arg_units=[None], return_unit=lambda x: x)
    from brian2.codegen.functions import add_implementations
    # code object name
    add_implementations(foo, codes={'weave': {}})
    assert set(foo.implementations.keys()) == set([WeaveCodeObject])
    del foo.implementations[WeaveCodeObject]
    # language name
    add_implementations(foo, codes={'cpp': {}})
    assert set(foo.implementations.keys()) == set([CPPLanguage])
    del foo.implementations[CPPLanguage]
    # class object
    add_implementations(foo, codes={CPPLanguage: {}})
    assert set(foo.implementations.keys()) == set([CPPLanguage])
    # unknown name
    assert_raises(ValueError, lambda: add_implementations(foo,
                                                          codes={'unknown': {}}))


def test_user_defined_function_discarding_units():
    # A function with units that should discard units also inside the function
    @make_function(discard_units=True)
    @check_units(v=volt, result=volt)
    def foo(v):
        return v + 3*volt  # this normally raises an error for unitless v

    assert foo(5*volt) == 8*volt

    # Test the function that is used during a run
    from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
    assert foo.implementations[NumpyCodeObject].code(5) == 8


def test_function_implementation_container():
    from brian2.core.functions import FunctionImplementationContainer
    import brian2.codegen.targets as targets

    class ALanguage(Language):
        language_id = 'A language'

    class BLanguage(Language):
        language_id = 'B language'

    class ACodeObject(CodeObject):
        language = ALanguage()
        class_name = 'A'

    class BCodeObject(CodeObject):
        language = BLanguage()
        class_name = 'B'

    # Register the code generation targets
    _previous_codegen_targets = set(targets.codegen_targets)
    targets.codegen_targets = set([ACodeObject, BCodeObject])

    container = FunctionImplementationContainer()

    # inserting into the container with a Language class
    container[BLanguage] = 'implementation B language'
    assert container[BLanguage] == 'implementation B language'

    # inserting into the container with a CodeObject class
    container[ACodeObject] = 'implementation A CodeObject'
    assert container[ACodeObject] == 'implementation A CodeObject'

    # does the fallback to the language work?
    assert container[BCodeObject] == 'implementation B language'

    assert_raises(KeyError, lambda: container['unknown'])

    # some basic dictionary properties
    assert len(container) == 2
    del container[ACodeObject]
    assert len(container) == 1
    assert set((key for key in container)) == set([BLanguage])

    # Restore the previous codegeneration targets
    targets.codegen_targets = _previous_codegen_targets


if __name__ == '__main__':
    test_math_functions()
    test_user_defined_function()
    test_simple_user_defined_function()
    test_manual_user_defined_function()
    test_user_defined_function_discarding_units()
    test_function_implementation_container()
