import numpy as np
from numpy.testing.utils import assert_raises

from brian2.core.namespace import create_namespace
from brian2.units import second, volt
from brian2.units.stdunits import ms, Hz, mV
from brian2.units.unitsafefunctions import sin, log, exp
from brian2.utils.logger import catch_logs

def _assert_one_warning(l):
    assert len(l) == 1, "expected one warning got %d" % len(l)
    assert l[0][0] == 'WARNING', "expected a WARNING, got %s instead" % l[0][0]


def test_default_content():
    '''
    Test that the default namespace contains standard units and functions.
    '''
    namespace = create_namespace({})
    # Units
    assert namespace['second'] == second
    assert namespace['volt'] == volt
    assert namespace['ms'] == ms
    assert namespace['Hz'] == Hz
    assert namespace['mV'] == mV
    # Functions (the namespace contains variables)
    assert namespace['sin'].pyfunc == sin 
    assert namespace['log'].pyfunc == log
    assert namespace['exp'].pyfunc == exp


def test_explicit_namespace():
    ''' Test resolution with an explicitly provided namespace '''
    
    explicit_namespace = {'variable': 'explicit_var'}
    # Explicitly provided 
    namespace = create_namespace( explicit_namespace)

    with catch_logs() as l:
        assert namespace['variable'] == 'explicit_var'
        assert len(l) == 0


def test_errors():
    # No explicit namespace
    namespace = create_namespace()
    assert_raises(KeyError, lambda: namespace['nonexisting_variable'])

    # Empty explicit namespace
    namespace = create_namespace({})
    assert_raises(KeyError, lambda: namespace['nonexisting_variable'])


def test_resolution():
    # implicit namespace
    tau = 10*ms
    namespace = create_namespace()
    additional_namespace = ('implicit-namespace', {'tau': tau})
    resolved = namespace.resolve_all(['tau', 'ms'], additional_namespace)
    assert len(resolved) == 2
    assert type(resolved) == type(dict())
    # make sure that the units are stripped
    assert resolved['tau'] == np.asarray(tau)
    assert resolved['ms'] == float(ms)

    # explicit namespace
    namespace = create_namespace({'tau': 20 * ms})

    resolved = namespace.resolve_all(['tau', 'ms'], additional_namespace)
    assert len(resolved) == 2
    assert resolved['tau'] == np.asarray(20 * ms)


if __name__ == '__main__':
    test_default_content()
    test_explicit_namespace()
    test_errors()
    test_resolution()
