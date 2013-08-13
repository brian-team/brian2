import numpy as np
from numpy.testing import assert_equal

from brian2 import *
from brian2.utils.logger import catch_logs

def test_math_functions():
    '''
    Test that math functions give the same result, regardless of whether used
    directly or in generated Python or C++ code.
    '''
    test_array = np.array([-1, -0.5, 0, 0.5, 1])
    
    # We can only test C++ if weave is availabe
    try:
        import scipy.weave
        codeobj_classes = [WeaveCodeObject, NumpyCodeObject]
    except ImportError:
        # Can't test C++
        codeobj_classes = [NumpyCodeObject]
    
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


if __name__ == '__main__':
    test_math_functions()
