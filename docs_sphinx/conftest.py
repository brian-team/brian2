# Do the equivalent of "from brian2 import *" for all doctests
import pytest

@pytest.fixture(autouse=True)
def add_brian2(doctest_namespace):
    exec('from brian2 import *', doctest_namespace)
    import numpy as np
    # Print output changed in numpy 1.14, stick with the old format to
    # avoid doctest failures
    try:
        np.set_printoptions(legacy='1.13')
    except TypeError:
        pass  # using a numpy version < 1.14
