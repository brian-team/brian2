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
    # Mismatching dtype information
    assert_raises(TypeError, lambda: Variable(Unit(1),
                                              value=np.arange(10),
                                              dtype=np.float32))
    # Boolean variable that isn't dimensionless
    assert_raises(ValueError, lambda: Variable(second,
                                               is_bool=True))

    # Dynamic array variable that is constant but not constant in size
    assert_raises(ValueError, lambda: DynamicArrayVariable('name', Unit(1),
                                                           dimensions=1,
                                                           value=None,
                                                           constant=True,
                                                           constant_size=False))


def test_str_repr():
    # Basic test that the str/repr methods work
    FakeGroup = namedtuple('G', ['name'])
    group = FakeGroup(name='groupname')
    variables = [Variable(second),
                 AuxiliaryVariable(second),
                 StochasticVariable(),
                 AttributeVariable(second, group, 'name'),
                 ArrayVariable('name', second,
                               value=None, group_name=group.name),
                 DynamicArrayVariable('name', second, dimensions=1,
                                      value=None, group_name=group.name),
                 Subexpression('sub', second, expr='a+b', group=group)]
    for var in variables:
        assert len(str(var))
        # The repr value should contain the name of the class
        assert len(repr(var)) and var.__class__.__name__ in repr(var)


if __name__ == '__main__':
    test_construction_errors()
    test_str_repr()
