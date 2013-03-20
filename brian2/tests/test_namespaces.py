import numpy as np
from numpy.testing.utils import assert_raises

from brian2.units import second, volt
from brian2.units.stdunits import ms, Hz, mV
from brian2.units.unitsafefunctions import sin, log, exp
from brian2.core.namespace import (ObjectWithNamespace)
from brian2.utils.logger import catch_logs

def _assert_one_warning(l):
    assert len(l) == 1, "expected one warning got %d" % len(l)
    assert l[0][0] == 'WARNING', "expected a WARNING, got %s instead" % l[0][0]


def test_default_content():
    '''
    Test that the default namespace contains standard units and functions.
    '''
    obj = ObjectWithNamespace()
    namespace = obj.create_namespace(1)
    # Units
    assert namespace['second'] == second
    assert namespace['volt'] == volt
    assert namespace['ms'] == ms
    assert namespace['Hz'] == Hz
    assert namespace['mV'] == mV
    # Functions
    assert namespace['sin'] == sin
    assert namespace['log'] == log
    assert namespace['exp'] == exp


def test_explicit_namespace():
    ''' Test resolution with an explicitly provided namespace '''
    
    obj = ObjectWithNamespace()
    explicit_namespace = {'variable': 'explicit_var',
                          'randn': 'explicit_randn'}
    # Explicitly provided 
    namespace = obj.create_namespace(1, explicit_namespace)
    
    
    with catch_logs() as l:
        assert namespace['variable'] == 'explicit_var'
        assert len(l) == 0
    
    with catch_logs() as l:
        # The explicitly provided namespace should take precedence over
        # the standard function namespace
        assert namespace['randn'] == 'explicit_randn'
        _assert_one_warning(l)


def test_implicit_namespace():
    ''' Test resolution with an implicitly provided namespace '''
    
    # import something into the local namespace
    from brian2.units.unitsafefunctions import sin
    
    variable = 'local_variable'
    
    obj = ObjectWithNamespace()
    
    # No explicitly provided namespace --> use locals and globals 
    namespace = obj.create_namespace(1)
    
    
    with catch_logs() as l:
        # no conflict here
        assert namespace['variable'] == 'local_variable'
        assert len(l) == 0
    
    with catch_logs() as l:
        assert namespace['sin'] == sin
        # There is a conflict here: sin is in the local namespace but also in
        # the default numpy namespace. We do *not* want to raise a warning here
        # however as both refer to the same thing
        assert len(l) == 0  


def test_errors():
    obj = ObjectWithNamespace()
    
    namespace = obj.create_namespace(1)
    
    assert_raises(KeyError, lambda: namespace['nonexisting_variable'])


def test_resolution():
    # implicit namespace
    tau = 10 * ms
    obj = ObjectWithNamespace()
    namespace = obj.create_namespace(1)

    resolved = namespace.resolve_all(['tau', 'ms'])
    assert len(resolved) == 2
    assert type(resolved) == type(dict())
    # make sure that the units are stripped
    assert resolved['tau'] == np.asarray(tau)
    assert resolved['ms'] == float(ms)

    # explicit namespace
    obj = ObjectWithNamespace()
    namespace = obj.create_namespace(1, {'tau': 20 * ms})

    resolved = namespace.resolve_all(['tau', 'ms'])
    assert len(resolved) == 2
    assert resolved['tau'] == np.asarray(20 * ms)


if __name__ == '__main__':
    test_default_content()
    test_explicit_namespace()
    test_implicit_namespace()
    test_errors()
    test_resolution()
