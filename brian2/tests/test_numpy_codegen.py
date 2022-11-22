import pytest

from brian2 import *


def test_error_message():
    if prefs.codegen.target != "numpy":
        pytest.skip("numpy-only test")

    @check_units(x=1, result=1)
    def foo(x):
        raise ValueError()

    G = NeuronGroup(1, "v : 1")
    G.run_regularly("v = foo(3)")
    with pytest.raises(BrianObjectException) as exc:
        run(defaultclock.dt)
        # The actual code line should be mentioned in the error message
        exc.match("v = foo(3)")


if __name__ == "__main__":
    prefs.codegen.target = "numpy"
    test_error_message()
