import numpy as np

from brian2.codegen.translation import analyse_identifiers, get_identifiers_recursively
from brian2.core.variables import Subexpression, Variable
from brian2.units.fundamentalunits import Unit


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
    variables = {'sub1': Subexpression(Unit(1), np.float32, 'sub2 * z'),
                 'sub2': Subexpression(Unit(1), np.float32, '5 + y'),
                 'x': Variable(unit=None)}
    identifiers = get_identifiers_recursively('_x = sub1 + x', variables)
    assert identifiers == set(['x', '_x', 'y', 'z', 'sub1', 'sub2'])

if __name__ == '__main__':
    test_analyse_identifiers()
    test_get_identifiers_recursively()
