import pytest
from brian2 import *
from brian2.core.base import BrianObjectException


@pytest.mark.codegen_independent
def test_shadowed_internal_variable_function_call_error():
    """
    Regression test for issue #1341: calling an internal variable (like 'xi')
    as a function should raise a helpful SyntaxError, not a ValueError.
    """
    start_scope()

    # Define an external variable named 'xi' (TimedArray)
    xi = TimedArray([1] * second**-0.5, dt=1 * ms)

    # We use 'xi(t)' in the equation.
    eqs = """dV/dt = (-V + sqrt(ms) * xi(t))/ms : 1"""

    group = NeuronGroup(1, eqs, method="euler")

    # Brian2 wraps the SyntaxError in a BrianObjectException during run()
    with pytest.raises(BrianObjectException) as excinfo:
        run(0 * ms)

    # Dig out the original SyntaxError from the wrapper
    original_error = excinfo.value.__cause__

    # Verify it is indeed a SyntaxError
    assert isinstance(original_error, SyntaxError)

    # Verify the message contains our helpful hint
    msg = str(original_error)
    assert "variable" in msg
    assert "used like a function" in msg
    assert "shadowing" in msg


if __name__ == "__main__":
    test_shadowed_internal_variable_function_call_error()
    print("Test passed!")
