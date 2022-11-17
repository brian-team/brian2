"""
Some basic tests for the `Variable` system
"""

from collections import namedtuple

import pytest
import numpy as np

from brian2.core.preferences import prefs
from brian2.core.variables import *
from brian2.units.fundamentalunits import Unit
from brian2.units.allunits import second


@pytest.mark.codegen_independent
def test_construction_errors():
    # Boolean variable that isn't dimensionless
    with pytest.raises(ValueError):
        Variable(name="name", dimensions=second.dim, dtype=bool)

    # Dynamic array variable that is constant but not constant in size
    with pytest.raises(ValueError):
        DynamicArrayVariable(
            name="name",
            owner=None,
            size=0,
            device=None,
            constant=True,
            needs_reference_update=True,
        )


@pytest.mark.codegen_independent
def test_str_repr():
    # Basic test that the str/repr methods work
    FakeGroup = namedtuple("G", ["name"])
    group = FakeGroup(name="groupname")
    variables = [
        Variable(name="name", dimensions=second.dim),
        Constant(name="name", dimensions=second.dim, value=1.0),
        AuxiliaryVariable(name="name", dimensions=second.dim),
        ArrayVariable(
            name="name", dimensions=second.dim, owner=None, size=10, device=None
        ),
        DynamicArrayVariable(
            name="name", dimensions=second.dim, owner=None, size=0, device=None
        ),
        Subexpression(
            name="sub", dimensions=second.dim, expr="a+b", owner=group, device=None
        ),
    ]
    for var in variables:
        assert len(str(var))
        # The repr value should contain the name of the class
        assert len(repr(var)) and var.__class__.__name__ in repr(var)


@pytest.mark.codegen_independent
def test_dtype_str():
    FakeGroup = namedtuple("G", ["name"])
    group = FakeGroup(name="groupname")
    for d in ["int32", "int64", "float32", "float64", "bool", "int", "float"]:
        nd = np.dtype(d)
        for var in [
            Constant(name="name", value=np.zeros(1, dtype=nd)[0]),
            AuxiliaryVariable(name="name", dtype=nd),
            ArrayVariable(name="name", owner=None, size=10, device=None, dtype=nd),
            DynamicArrayVariable(
                name="name", owner=None, dtype=nd, size=0, device=None
            ),
            Subexpression(name="sub", expr="a+b", owner=group, device=None, dtype=nd),
        ]:
            assert var.dtype_str.startswith(d)


if __name__ == "__main__":
    test_construction_errors()
    test_str_repr()
    test_dtype_str()
