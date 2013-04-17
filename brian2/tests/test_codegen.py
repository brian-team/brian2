import numpy as np
from numpy.testing import assert_raises, assert_equal

from brian2.codegen.translation import analyse_identifiers

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


if __name__ == '__main__':
    test_analyse_identifiers()
