import numpy as np
import pytest
from numpy.testing import assert_equal

from brian2.core.magic import run
from brian2.devices.device import (
    Device,
    RuntimeDevice,
    all_devices,
    get_device,
    reset_device,
    runtime_device,
    set_device,
)
from brian2.groups.neurongroup import NeuronGroup
from brian2.units import ms


class ATestDevice(Device):
    def activate(self, build_on_run, **kwargs):
        super().activate(build_on_run, **kwargs)
        self.build_on_run = build_on_run
        self._options = kwargs

    # These functions are needed during the setup of the defaultclock
    def get_value(self, var):
        return np.array([0.0001])

    def add_array(self, var):
        pass

    def init_with_zeros(self, var, dtype):
        pass

    def fill_with_array(self, var, arr):
        pass


@pytest.mark.codegen_independent
def test_set_reset_device_implicit():
    from brian2.devices import device_module

    old_prev_devices = list(device_module.previous_devices)
    device_module.previous_devices = []
    test_device1 = ATestDevice()
    all_devices["test1"] = test_device1
    test_device2 = ATestDevice()
    all_devices["test2"] = test_device2

    set_device("test1", build_on_run=False, my_opt=1)
    set_device("test2", build_on_run=True, my_opt=2)
    assert get_device() is test_device2
    assert get_device()._options["my_opt"] == 2
    assert get_device().build_on_run

    reset_device()
    assert get_device() is test_device1
    assert get_device()._options["my_opt"] == 1
    assert not get_device().build_on_run

    reset_device()
    assert get_device() is runtime_device

    reset_device()  # If there is no previous device, will reset to runtime device
    assert get_device() is runtime_device
    del all_devices["test1"]
    del all_devices["test2"]
    device_module.previous_devices = old_prev_devices


@pytest.mark.codegen_independent
def test_set_reset_device_explicit():
    original_device = get_device()
    test_device1 = ATestDevice()
    all_devices["test1"] = test_device1
    test_device2 = ATestDevice()
    all_devices["test2"] = test_device2
    test_device3 = ATestDevice()
    all_devices["test3"] = test_device3

    set_device("test1", build_on_run=False, my_opt=1)
    set_device("test2", build_on_run=True, my_opt=2)
    set_device("test3", build_on_run=False, my_opt=3)

    reset_device("test1")  # Directly jump back to the first device
    assert get_device() is test_device1
    assert get_device()._options["my_opt"] == 1
    assert not get_device().build_on_run

    del all_devices["test1"]
    del all_devices["test2"]
    del all_devices["test3"]
    reset_device(original_device)


@pytest.mark.skipif(
    not isinstance(get_device(), RuntimeDevice),
    reason="Getting/setting random number state only supported for runtime device.",
)
def test_get_set_random_generator_state():
    group = NeuronGroup(10, "dv/dt = -v/(10*ms) + (10*ms)**-0.5*xi : 1", method="euler")
    group.v = "rand()"
    run(10 * ms)
    assert np.var(group.v) > 0  # very basic test for randomness ;)
    old_v = np.array(group.v)
    random_state = get_device().get_random_state()
    group.v = "rand()"
    run(10 * ms)
    assert np.var(group.v - old_v) > 0  # just checking for *some* difference
    old_v = np.array(group.v)
    get_device().set_random_state(random_state)
    group.v = "rand()"
    run(10 * ms)
    assert_equal(group.v, old_v)


if __name__ == "__main__":
    test_set_reset_device_implicit()
    test_set_reset_device_explicit()
    test_get_set_random_generator_state()
