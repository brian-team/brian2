import numpy as np
from numpy.testing.utils import assert_raises, assert_equal, assert_allclose

from brian2 import *


def test_state_variables():
    '''
    Test the setting and accessing of state variables in subgroups.
    '''
    G = NeuronGroup(10, 'v : volt')
    SG = G[4:9]
    assert_raises(DimensionMismatchError, lambda: SG.__setattr__('v', -70))
    SG.v_ = float(-80*mV)
    assert_allclose(G.v,
                    np.array([0, 0, 0, 0, -80, -80, -80, -80, -80, 0])*mV)
    assert_allclose(SG.v,
                    np.array([-80, -80, -80, -80, -80])*mV)
    assert_allclose(G.v_,
                    np.array([0, 0, 0, 0, -80, -80, -80, -80, -80, 0])*mV)
    assert_allclose(SG.v_,
                    np.array([-80, -80, -80, -80, -80])*mV)
    # You should also be able to set variables with a string
    SG.v = 'v + i*mV'
    assert_allclose(SG.v[0], -80*mV)
    assert_allclose(SG.v[4], -76*mV)
    assert_allclose(G.v[4:9], -80*mV + np.arange(5)*mV)

    # Calculating with state variables should work too
    assert all(G.v[4:9] - SG.v == 0)

    # And in-place modification should work as well
    SG.v += 10*mV
    SG.v *= 2
    # with unit checking
    assert_raises(DimensionMismatchError, lambda: SG.v.__iadd__(3*second))
    assert_raises(DimensionMismatchError, lambda: SG.v.__iadd__(3))
    assert_raises(DimensionMismatchError, lambda: SG.v.__imul__(3*second))


if __name__ == '__main__':
    test_state_variables()