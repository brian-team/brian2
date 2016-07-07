from nose.plugins.skip import SkipTest

from brian2 import *

def test_error_message():
    if prefs.codegen.target != 'numpy':
        raise SkipTest('numpy-only test')

    @check_units(x=1, result=1)
    def foo(x):
        raise ValueError()

    G = NeuronGroup(1, 'v : 1')
    G.run_regularly('v = foo(3)')
    try:
        run(defaultclock.dt)
        raise AssertionError('Expected the run to raise a ValueError')
    except ValueError as exc:
        # The actual code line should be mentioned in the error message
        assert 'v = foo(3)' in str(exc)


if __name__ == '__main__':
    prefs.codegen.target = 'numpy'
    test_error_message()
