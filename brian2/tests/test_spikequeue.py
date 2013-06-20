import numpy as np
from numpy.testing.utils import assert_equal
from brian2.synapses.spikequeue import SpikeQueue
from brian2.units.stdunits import ms
from memory.dynamicarray import DynamicArray1D


def create_all_to_all(N):
    '''
    Return a tuple containing `synapses` and `delays` in the form that is needed
    for the `SpikeQueue` initializer.
    Every synapse has a delay depending on the presynaptic neuron.
    '''
    data = np.repeat(np.arange(N), N)
    delays = DynamicArray1D(data.shape, dtype=int)
    delays[:] = data
    data = np.repeat(np.arange(N), N)
    synapses = DynamicArray1D(data.shape, dtype=int)
    synapses[:] = data
    return synapses, delays


def create_one_to_one(N):
    '''
    Return a tuple containing `synapses` and `delays` in the form that is needed
    for the `SpikeQueue` initializer.
    Every synapse has a delay depending on the presynaptic neuron.
    '''
    data = np.arange(N)
    delays = DynamicArray1D(data.shape, dtype=int)
    delays[:] = data
    data = np.arange(N)
    synapses = DynamicArray1D(data.shape, dtype=int)
    synapses[:] = data
    return synapses, delays


def test_spikequeue():
    N = 100
    synapses, delays = create_one_to_one(N)
    queue = SpikeQueue(synapses, delays, 0.1*ms)
    queue.compress()
    queue.push(np.arange(N))
    for i in xrange(N):
        assert_equal(queue.peek(), np.array([i]))
        queue.next()
    for i in xrange(N):
        assert_equal(queue.peek(), np.array([]))
        queue.next()

    synapses, delays = create_all_to_all(N)
    queue = SpikeQueue(synapses, delays, 0.1*ms)
    queue.compress()
    queue.push(np.arange(N))
    for i in xrange(N):
        assert_equal(queue.peek(), i*N + np.arange(N))
        queue.next()
    for i in xrange(N):
        assert_equal(queue.peek(), np.array([]))
        queue.next()


if __name__ == '__main__':
    test_spikequeue()
