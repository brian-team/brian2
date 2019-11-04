import re

import numpy as np
import pytest

from brian2.units import ms
from brian2.core.clocks import defaultclock
from brian2.core.functions import Function, DEFAULT_FUNCTIONS
from brian2.devices.device import reinit_and_delete


def pytest_ignore_collect(path, config):
    if config.option.doctestmodules:
        if 'tests' in str(path):
            return True  # Ignore tests package for doctests
    # Do not test brian2.hears bridge (needs Brian1)
    if str(path).endswith('hears.py'):
        return True


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

# Fixture that is used for all tests
@pytest.fixture(autouse=True)
def setup_and_teardown(request):
    # Set preferences before each test
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
    # Print output changed in numpy 1.14, stick with the old format to
    # avoid doctest failures
    try:
        np.set_printoptions(legacy='1.13')
    except TypeError:
        pass  # using a numpy version < 1.14

    yield  # run test

    # clean up after the test
    reinit_and_delete()
    # Reset defaultclock.dt to be sure
    defaultclock.dt = 0.1*ms


# (Optionally) mark tests raising NotImplementedError as skipped (mostly used
# for testing Brian2GeNN)
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    if hasattr(item.config, 'slaveinput'):
        fail_for_not_implemented = item.config.slaveinput['fail_for_not_implemented']
    else:
        fail_for_not_implemented = item.config.fail_for_not_implemented
    outcome = yield
    rep = outcome.get_result()
    if not fail_for_not_implemented and rep.outcome == 'failed':
        if call.excinfo.errisinstance(NotImplementedError):
            rep.outcome = 'skipped'
            r = call.excinfo._getreprcrash()
            rep.longrepr = (str(r.path), r.lineno, r.message)
