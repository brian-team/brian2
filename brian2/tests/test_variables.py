'''
Some basic tests for the `Variable` system
'''
from collections import namedtuple

from nose.plugins.attrib import attr
import numpy as np
from numpy.testing import assert_raises

from brian2.core.preferences import prefs
from brian2.core.variables import *
from brian2.units.fundamentalunits import Unit
from brian2.units.allunits import second


@attr('codegen-independent')
def test_construction_errors():
    # Boolean variable that isn't dimensionless
    assert_raises(ValueError, lambda: Variable(name='name', unit=second,
                                               dtype=np.bool))

    # Dynamic array variable that is constant but not constant in size
    assert_raises(ValueError, lambda: DynamicArrayVariable(name='name',
                                                           unit=Unit(1),
                                                           owner=None,
                                                           size=0,
                                                           device=None,
                                                           constant=True,
                                                           needs_reference_update=True))


@attr('codegen-independent')
def test_str_repr():
    # Basic test that the str/repr methods work
    FakeGroup = namedtuple('G', ['name'])
    group = FakeGroup(name='groupname')
    variables = [Variable(name='name', unit=second),
                 Constant(name='name', unit=second, value=1.0),
                 AuxiliaryVariable(name='name', unit=second),
                 ArrayVariable(name='name', unit=second, owner=None, size=10, device=None),
                 DynamicArrayVariable(name='name', unit=second, owner=None, size=0,
                                      device=None),
                 Subexpression(name='sub', unit=second, expr='a+b', owner=group,
                               device=None)]
    for var in variables:
        assert len(str(var))
        # The repr value should contain the name of the class
        assert len(repr(var)) and var.__class__.__name__ in repr(var)


@attr('codegen-independent')
def test_dtype_str():
    FakeGroup = namedtuple('G', ['name'])
    group = FakeGroup(name='groupname')
    for d in ['int32', 'int64', 'float32', 'float64', 'bool', 'int', 'float']:
        nd = np.dtype(d)
        for var in [Constant(name='name', unit=1,
                             value=np.zeros(1, dtype=nd)[0]),
                    AuxiliaryVariable(name='name', dtype=nd, unit=1),
                    ArrayVariable(name='name', owner=None, size=10, unit=1,
                                  device=None, dtype=nd),
                    DynamicArrayVariable(name='name', owner=None, dtype=nd,
                                         size=0, device=None, unit=1),
                    Subexpression(name='sub', expr='a+b', owner=group, unit=1,
                                  device=None, dtype=nd)]:
            assert var.dtype_str.startswith(d)


if __name__ == '__main__':
    test_construction_errors()
    test_str_repr()
    test_dtype_str()
