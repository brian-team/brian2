import numpy as np
from nose.plugins.attrib import attr

from brian2.devices.device import (Device, all_devices, set_device, get_device,
                                   reset_device, runtime_device, previous_devices)

class ATestDevice(Device):
    def activate(self, build_on_run, **kwargs):
        super(ATestDevice, self).activate(build_on_run, **kwargs)
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

@attr('codegen-independent')
def test_set_reset_device_implicit():

    test_device1 = ATestDevice()
    all_devices['test1'] = test_device1
    test_device2 = ATestDevice()
    all_devices['test2'] = test_device2

    set_device('test1', build_on_run=False, my_opt=1)
    set_device('test2', build_on_run=True, my_opt=2)
    assert get_device() is test_device2
    assert get_device()._options['my_opt'] == 2
    assert get_device().build_on_run

    reset_device()
    assert get_device() is test_device1
    assert get_device()._options['my_opt'] == 1
    assert not get_device().build_on_run

    reset_device()
    assert get_device() is runtime_device

    reset_device()  # If there is no previous device, will reset to runtime device
    assert get_device() is runtime_device
    del all_devices['test1']
    del all_devices['test2']

@attr('codegen-independent')
def test_set_reset_device_explicit():
    original_device = get_device()
    test_device1 = ATestDevice()
    all_devices['test1'] = test_device1
    test_device2 = ATestDevice()
    all_devices['test2'] = test_device2
    test_device3 = ATestDevice()
    all_devices['test3'] = test_device3

    set_device('test1', build_on_run=False, my_opt=1)
    set_device('test2', build_on_run=True, my_opt=2)
    set_device('test3', build_on_run=False, my_opt=3)

    reset_device('test1')  # Directly jump back to the first device
    assert get_device() is test_device1
    assert get_device()._options['my_opt'] == 1
    assert not get_device().build_on_run

    del all_devices['test1']
    del all_devices['test2']
    del all_devices['test3']
    reset_device(original_device)

if __name__ == '__main__':
    test_set_reset_device_implicit()
    test_set_reset_device_explicit()
