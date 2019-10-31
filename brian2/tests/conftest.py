import pytest

import re
from io import StringIO

import numpy as np

from brian2.units import ms
from brian2.core.clocks import defaultclock
from brian2.core.functions import Function, DEFAULT_FUNCTIONS
from brian2.devices.device import reinit_and_delete

@pytest.fixture
def device_teardown():
    # This fixture is not doing anything before the test
    yield None
    # clean up after the test
    reinit_and_delete()

@pytest.fixture
def restore_clock_teardown():
    # This fixture is not doing anything before the test
    yield None
    defaultclock.dt = 0.1 * ms


# The "random" values are always 0.5
def fake_randn(vectorisation_idx):
    return 0.5*np.ones_like(vectorisation_idx)
fake_randn = Function(fake_randn, arg_units=[], return_unit=1, auto_vectorise=True,
                      stateless=False)
fake_randn.implementations.add_implementation('cpp', '''
                                              double randn(int vectorisation_idx)
                                              {
                                                  return 0.5;
                                              }
                                              ''')
fake_randn.implementations.add_implementation('cython','''
                                    cdef double randn(int vectorisation_idx):
                                        return 0.5
                                    ''')
@pytest.fixture
def fake_randn_randn_fixture():
    orig_randn = DEFAULT_FUNCTIONS['randn']
    DEFAULT_FUNCTIONS['randn'] = fake_randn
    yield None
    DEFAULT_FUNCTIONS['randn'] = orig_randn

@pytest.fixture(autouse=True)
def set_preferences_fixture(request):
    import brian2
    if hasattr(request.config, 'slaveinput'):
        for key, value in request.config.slaveinput['brian_prefs'].items():
            if isinstance(value, tuple) and value[0] == 'TYPE':
                matches = re.match(r"<(type|class) 'numpy\.(.+)'>", value[1])
                if matches is None or len(matches.groups()) != 2:
                    raise TypeError('Do not know how to handle {} in '
                                    'preferences'.format(value[1]))
                t = matches.groups()[1]
                if t == 'float64':
                    value = np.float64
                elif t == 'float32':
                    value = np.float32
                elif t == 'int64':
                    value = np.int64
                elif t == 'int32':
                    value = np.int32

            brian2.prefs[key] = value
    else:
        for k, v in request.config.brian_prefs.items():
            brian2.prefs[k] = v

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    # TODO: make this optional
    rep = outcome.get_result()
    if rep.outcome == 'failed':
        if call.excinfo.errisinstance(NotImplementedError):
            rep.outcome = 'skipped'
            r = call.excinfo._getreprcrash()
            rep.longrepr = (str(r.path), r.lineno, r.message)
