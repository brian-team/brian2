from collections import namedtuple

import numpy as np
from numpy.testing import assert_raises

from brian2.codegen.translation import (analyse_identifiers,
                                        get_identifiers_recursively,
                                        make_statements,
                                        )
from brian2.core.variables import Subexpression, Variable
from brian2.units.fundamentalunits import Unit

FakeGroup = namedtuple('FakeGroup', ['variables'])

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
    
    assert defined==set(['a'])
    assert used_known==set(['b', 'c', 'd'])
    assert dependent==set(['e', 'f'])


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
    identifiers = get_identifiers_recursively('_x = sub1 + x', variables)
    assert identifiers == set(['x', '_x', 'y', 'z', 'sub1', 'sub2'])


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
    stmts = make_statements(code, variables, np.float32)
    evalorder = ''.join(stmt.var for stmt in stmts)
    # This is the order that variables ought to be evaluated in
    assert evalorder=='baxcbaxdax'
    

if __name__ == '__main__':
    test_analyse_identifiers()
    test_get_identifiers_recursively()
    test_nested_subexpressions()
