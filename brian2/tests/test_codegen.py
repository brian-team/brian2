from collections import namedtuple

import numpy as np
from numpy.testing import assert_raises
from nose.plugins.attrib import attr

from brian2.codegen.translation import (analyse_identifiers,
                                        get_identifiers_recursively,
                                        make_statements,
                                        apply_loop_invariant_optimisations
                                        )
from brian2.codegen.statements import Statement
from brian2.codegen.codeobject import CodeObject
from brian2.core.variables import Subexpression, Variable, Constant
from brian2.core.functions import Function, DEFAULT_FUNCTIONS
from brian2.devices.device import auto_target
from brian2.units.fundamentalunits import Unit
from brian2.units import second, ms

FakeGroup = namedtuple('FakeGroup', ['variables'])

@attr('codegen-independent')
def test_auto_target():
    # very basic test that the "auto" codegen target is useable
    assert issubclass(auto_target(), CodeObject)


@attr('codegen-independent')
def test_analyse_identifiers():
    '''
    Test that the analyse_identifiers function works on a simple clear example.
    '''
    code = '''
    a = b+c
    d = e+f
    '''
    known = {'b': Variable(unit=None, name='b'),
             'c': Variable(unit=None, name='c'),
             'd': Variable(unit=None, name='d'),
             'g': Variable(unit=None, name='g')}
    
    defined, used_known, dependent = analyse_identifiers(code, known)
    assert 'a' in defined  # There might be an additional constant added by the
                           # loop-invariant optimisation
    assert used_known == {'b', 'c', 'd'}
    assert dependent == {'e', 'f'}


@attr('codegen-independent')
def test_get_identifiers_recursively():
    '''
    Test finding identifiers including subexpressions.
    '''
    variables = {'sub1': Subexpression(name='sub1', unit=Unit(1),
                                       dtype=np.float32, expr='sub2 * z',
                                       owner=FakeGroup(variables={}),
                                       device=None),
                 'sub2': Subexpression(name='sub2', unit=Unit(1),
                                       dtype=np.float32, expr='5 + y',
                                       owner=FakeGroup(variables={}),
                                       device=None),
                 'x': Variable(unit=None, name='x')}
    identifiers = get_identifiers_recursively(['_x = sub1 + x'],
                                              variables)
    assert identifiers == {'x', '_x', 'y', 'z', 'sub1', 'sub2'}


@attr('codegen-independent')
def test_nested_subexpressions():
    '''
    This test checks that code translation works with nested subexpressions.
    '''
    code = '''
    x = a + b + c
    c = 1
    x = a + b + c
    d = 1
    x = a + b + c
    '''
    variables = {
        'a': Subexpression(name='a', unit=Unit(1), dtype=np.float32, owner=FakeGroup(variables={}), device=None,
                           expr='b*b+d'),
        'b': Subexpression(name='b', unit=Unit(1), dtype=np.float32, owner=FakeGroup(variables={}), device=None,
                           expr='c*c*c'),
        'c': Variable(unit=None, name='c'),
        'd': Variable(unit=None, name='d'),
        }
    scalar_stmts, vector_stmts = make_statements(code, variables, np.float32)
    assert len(scalar_stmts) == 0
    evalorder = ''.join(stmt.var for stmt in vector_stmts)
    # This is the order that variables ought to be evaluated in
    assert evalorder=='baxcbaxdax'
    
@attr('codegen-independent')
def test_apply_loop_invariant_optimisation():
    variables = {'v': Variable('v', Unit(1), scalar=False),
                 'w': Variable('w', Unit(1), scalar=False),
                 'dt': Constant('dt', second, 0.1*ms),
                 'tau': Constant('tau', second, 10*ms),
                 'exp': DEFAULT_FUNCTIONS['exp']}
    statements = [Statement('v', '=', 'dt*w*exp(-dt/tau)/tau + v*exp(-dt/tau)', '', np.float32),
                  Statement('w', '=', 'w*exp(-dt/tau)', '', np.float32)]
    scalar, vector = apply_loop_invariant_optimisations(statements, variables,
                                                        np.float64)
    # The optimisation should pull out exp(-dt / tau)
    assert len(scalar) == 1
    assert scalar[0].dtype == np.float64  # We asked for this dtype above
    assert scalar[0].var == '_lio_const_1'
    assert len(vector) == 2
    assert all('_lio_const_1' in stmt.expr for stmt in vector)


@attr('codegen-independent')
def test_apply_loop_invariant_optimisation_integer():
    variables = {'v': Variable('v', Unit(1), scalar=False),
                 'N': Constant('N', Unit(1), 10)}
    statements = [Statement('v', '=', 'v % (2*3*N)', '', np.float32)]
    scalar, vector = apply_loop_invariant_optimisations(statements, variables,
                                                        np.float64)
    # The optimisation should not pull out 2*N
    assert len(scalar) == 0

if __name__ == '__main__':
    test_auto_target()
    test_analyse_identifiers()
    test_get_identifiers_recursively()
    test_nested_subexpressions()
    test_apply_loop_invariant_optimisation()
    test_apply_loop_invariant_optimisation_integer()