from collections import namedtuple

import numpy as np
from numpy.testing import assert_raises

from brian2.codegen.translation import (analyse_identifiers,
                                        get_identifiers_recursively,
                                        translate_subexpression)
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
    known = ['b', 'c', 'd', 'g']
    
    defined, used_known, dependent = analyse_identifiers(code, known)
    
    assert defined==set(['a'])
    assert used_known==set(['b', 'c', 'd'])
    assert dependent==set(['e', 'f'])


def test_get_identifiers_recursively():
    '''
    Test finding identifiers including subexpressions.
    '''
    variables = {'sub1': Subexpression('sub1', Unit(1), dtype=np.float32,
                                       expr='sub2 * z',
                                       group=FakeGroup(variables={})),
                 'sub2': Subexpression('sub2', Unit(1), dtype=np.float32,
                                       expr='5 + y',
                                       group=FakeGroup(variables={})),
                 'x': Variable(unit=None)}
    identifiers = get_identifiers_recursively('_x = sub1 + x', variables)
    assert identifiers == set(['x', '_x', 'y', 'z', 'sub1', 'sub2'])


def test_translate_subexpression():
    var1, var2 = object(), object()
    G1 = FakeGroup(variables={'var1': var1, 'var2': var2})
    G2 = FakeGroup(variables={'var1': var1, 'var2_post': var2})
    G3 = FakeGroup(variables={'var1': var1})
    sub = Subexpression('sub', Unit(1), dtype=None,
                        expr='var1 + var2', group=G1)
    # Interpreted in the context of G1, the variable names should stay the
    # same
    assert translate_subexpression(sub, G1.variables) == 'var1 + var2'
    # Interpreted in the context of G2, var2 should be translated
    assert translate_subexpression(sub, G2.variables) == 'var1 + var2_post'
    # Trying to interpret the subexpression in the context of G3 should raise
    # an error
    assert_raises(KeyError, lambda: translate_subexpression(sub, G3.variables))

if __name__ == '__main__':
    test_analyse_identifiers()
    test_get_identifiers_recursively()
    test_translate_subexpression()
