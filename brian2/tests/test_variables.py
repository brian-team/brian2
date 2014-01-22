'''
Some basic tests for the `Variable` system
'''
from collections import namedtuple
import numpy as np
from numpy.testing import assert_raises

from brian2.core.variables import *
from brian2.units.fundamentalunits import Unit
from brian2.units.allunits import second


def test_construction_errors():
    # Boolean variable that isn't dimensionless
    assert_raises(ValueError, lambda: Variable(name='name', unit=second,
                                               is_bool=True))

    # Dynamic array variable that is constant but not constant in size
    assert_raises(ValueError, lambda: DynamicArrayVariable(name='name',
                                                           unit=Unit(1),
                                                           owner=None,
                                                           size=0,
                                                           device=None,
                                                           constant=True,
                                                           constant_size=False))


def test_str_repr():
    # Basic test that the str/repr methods work
    FakeGroup = namedtuple('G', ['name'])
    group = FakeGroup(name='groupname')
    variables = [Variable(name='name', unit=second),
                 Constant(name='name', unit=second, value=1.0),
                 AuxiliaryVariable(name='name', unit=second),
                 AttributeVariable(name='name', unit=second, obj=group,
                                   attribute='name', dtype=np.float32),
                 ArrayVariable(name='name', unit=second, owner=None, size=10, device=None),
                 DynamicArrayVariable(name='name', unit=second, owner=None, size=0,
                                      device=None),
                 Subexpression(name='sub', unit=second, expr='a+b', owner=group,
                               device=None)]
    for var in variables:
        assert len(str(var))
        # The repr value should contain the name of the class
        assert len(repr(var)) and var.__class__.__name__ in repr(var)


if __name__ == '__main__':
    test_construction_errors()
    test_str_repr()
