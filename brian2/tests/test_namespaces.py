from collections import namedtuple

import numpy as np
from numpy.testing.utils import assert_raises

from brian2.core.namespace import get_local_namespace, resolve, resolve_all
from brian2.units import second, volt
from brian2.units.stdunits import ms, Hz, mV
from brian2.units.unitsafefunctions import sin, log, exp
from brian2.utils.logger import catch_logs

# A simple group class with a variables and a namespace argument
SimpleGroup = namedtuple('SimpleGroup', ['namespace', 'variables'])

def _assert_one_warning(l):
    assert len(l) == 1, "expected one warning got %d" % len(l)
    assert l[0][0] == 'WARNING', "expected a WARNING, got %s instead" % l[0][0]


def test_default_content():
    '''
    Test that the default namespace contains standard units and functions.
    '''
    # Units
    assert resolve('second', None) == second
    assert resolve('volt', None) == volt
    assert resolve('ms', None) == ms
    assert resolve('Hz', None) == Hz
    assert resolve('mV', None) == mV

    # Functions
    assert resolve('sin', None).pyfunc == sin
    assert resolve('log', None).pyfunc == log
    assert resolve('exp', None).pyfunc == exp


def test_explicit_namespace():
    ''' Test resolution with an explicitly provided namespace '''

    group = SimpleGroup(namespace={'variable': 42}, variables={})

    # Explicitly provided
    with catch_logs() as l:
        assert resolve('variable', group) == 42
        assert len(l) == 0


def test_errors():
    # No explicit namespace
    group = SimpleGroup(namespace=None, variables={})
    assert_raises(KeyError, lambda: resolve('nonexisting_variable', group))

    # Empty explicit namespace
    group = SimpleGroup(namespace={}, variables={})
    assert_raises(KeyError, lambda: resolve('nonexisting_variable', group))


def test_resolution():
    # implicit namespace
    tau = 10*ms
    group = SimpleGroup(namespace=None, variables={})
    resolved = resolve_all(['tau', 'ms'], group)
    assert len(resolved) == 2
    assert type(resolved) == type(dict())
    assert resolved['tau'] == tau
    assert resolved['ms'] == ms

    # explicit namespace
    group = SimpleGroup(namespace={'tau': 20 * ms}, variables={})

    resolved = resolve_all(['tau', 'ms'], group)
    assert len(resolved) == 2
    assert resolved['tau'] == 20 * ms


def test_warning():
    from brian2.core.functions import DEFAULT_FUNCTIONS
    from brian2.units.stdunits import cm as brian_cm
    # Name in external namespace clashes with unit/function name
    exp = 23
    cm = 42
    group = SimpleGroup(namespace=None, variables={})
    with catch_logs() as l:
        resolved = resolve('exp', group)
        assert resolved == DEFAULT_FUNCTIONS['exp']
        assert len(l) == 1
        assert l[0][1].endswith('.resolution_conflict')
    with catch_logs() as l:
        resolved = resolve('cm', group)
        assert resolved == brian_cm
        assert len(l) == 1
        assert l[0][1].endswith('.resolution_conflict')


if __name__ == '__main__':
    test_default_content()
    test_explicit_namespace()
    test_errors()
    test_resolution()
    test_warning()
