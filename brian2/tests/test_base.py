import pytest
from numpy.testing import assert_equal
from packaging.version import parse as parse_version

from brian2 import *
from brian2.devices.device import reinit_and_delete
from brian2.tests.utils import assert_allclose


class DerivedBrianObject(BrianObject):
    def __init__(self, name="derivedbrianobject*"):
        super().__init__(name=name)

    def __str__(self):
        return self.name

    __repr__ = __str__


@pytest.mark.codegen_independent
def test_base():
    x = DerivedBrianObject("x")
    y = DerivedBrianObject("y")
    assert_equal(x.when, "start")
    assert_equal(x.order, 0)
    assert_equal(len(x.contained_objects), 0)
    with pytest.raises(AttributeError):
        setattr(x, "contained_objects", [])
    x.contained_objects.append(y)
    assert_equal(len(x.contained_objects), 1)
    assert x.contained_objects[0] is y

    assert_equal(x.active, True)
    assert_equal(y.active, True)
    y.active = False
    assert_equal(x.active, True)
    assert_equal(y.active, False)
    y.active = True
    assert_equal(x.active, True)
    assert_equal(y.active, True)
    x.active = False
    assert_equal(x.active, False)
    assert_equal(y.active, False)


@pytest.mark.codegen_independent
def test_names():
    obj = BrianObject()
    obj2 = BrianObject()
    obj3 = DerivedBrianObject()
    assert_equal(obj.name, "brianobject")
    assert_equal(obj2.name, "brianobject_1")
    assert_equal(obj3.name, "derivedbrianobject")


@pytest.mark.codegen_independent
def test_duplicate_names():
    # duplicate names are allowed, as long as they are not part of the
    # same network
    obj1 = BrianObject(name="name1")
    obj2 = BrianObject(name="name2")
    obj3 = BrianObject(name="name")
    obj4 = BrianObject(name="name")
    net = Network(obj1, obj2)
    # all is good
    net.run(0 * ms)
    net = Network(obj3, obj4)
    with pytest.raises(ValueError):
        net.run(0 * ms)


@pytest.mark.standalone_compatible
@pytest.mark.multiple_runs
def test_active_flag():
    G = NeuronGroup(1, "dv/dt = 1/ms : 1")
    mon = StateMonitor(G, "v", record=0)
    mon.active = False
    run(1 * ms)
    mon.active = True
    G.active = False
    run(1 * ms)
    device.build(direct_call=False, **device.build_options)
    # Monitor should start recording at 1ms
    # Neurongroup should not integrate after 1ms (but should have integrated before)
    assert_allclose(mon[0].t[0], 1 * ms)
    assert_allclose(mon[0].v, 1.0)


@pytest.mark.codegen_independent
def test_version():
    # Check that determining the Brian version works correctly
    import brian2

    version = brian2.__version__
    assert version.startswith("2.")

    # Check that the version tuple is correct
    version_tuple = brian2.__version_tuple__

    expected = []
    for v in version.split("."):
        try:
            expected.append(int(v))
        except ValueError:
            expected.append(v)

    # Note that setuptools_scm >= v8.2.1 include a post suffix in the
    # version tutple, but earlier versions don't
    assert version_tuple == tuple(expected) or version_tuple == tuple(
        t for t in expected if isinstance(t, int)
    )


if __name__ == "__main__":
    test_base()
    test_names()
    test_duplicate_names()
    test_active_flag()
