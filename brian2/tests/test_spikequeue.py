import numpy as np
from numpy.testing import assert_equal
import pytest

from brian2.synapses.spikequeue import SpikeQueue
from brian2.units.stdunits import ms
from brian2.memory.dynamicarray import DynamicArray1D


def create_all_to_all(N, dt):
    """
    Return a tuple containing `synapses` and `delays` in the form that is needed
    for the `SpikeQueue` initializer.
    Every synapse has a delay depending on the presynaptic neuron.
    """
    data = np.repeat(np.arange(N, dtype=np.int32), N)
    delays = DynamicArray1D(data.shape, dtype=np.float64)
    delays[:] = data * dt
    synapses = data
    return synapses, delays


def create_one_to_one(N, dt):
    """
    Return a tuple containing `synapses` and `delays` in the form that is needed
    for the `SpikeQueue` initializer.
    Every synapse has a delay depending on the presynaptic neuron.
    """
    data = np.arange(N, dtype=np.int32)
    delays = DynamicArray1D(data.shape, dtype=np.float64)
    delays[:] = data * dt
    synapses = data
    return synapses, delays


@pytest.mark.codegen_independent
def test_spikequeue():
    N = 100
    dt = float(0.1 * ms)
    synapses, delays = create_one_to_one(N, dt)
    queue = SpikeQueue(source_start=0, source_end=N)
    queue.prepare(delays[:], dt, synapses)
    queue.push(np.arange(N, dtype=np.int32))
    for i in range(N):
        assert_equal(queue.peek(), np.array([i]))
        queue.advance()
    for i in range(N):
        assert_equal(queue.peek(), np.array([]))
        queue.advance()

    synapses, delays = create_all_to_all(N, dt)

    queue = SpikeQueue(source_start=0, source_end=N)
    queue.prepare(delays[:], dt, synapses)
    queue.push(np.arange(N * N, dtype=np.int32))
    for i in range(N):
        assert_equal(queue.peek(), i * N + np.arange(N))
        queue.advance()
    for i in range(N):
        assert_equal(queue.peek(), np.array([]))
        queue.advance()


if __name__ == "__main__":
    test_spikequeue()
