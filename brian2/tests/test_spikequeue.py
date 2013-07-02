import numpy as np
from numpy.testing.utils import assert_equal
from brian2.synapses.spikequeue import SpikeQueue
from brian2.units.stdunits import ms
from brian2.memory.dynamicarray import DynamicArray1D


def create_all_to_all(N):
    '''
    Return a tuple containing `synapses` and `delays` in the form that is needed
    for the `SpikeQueue` initializer.
    Every synapse has a delay depending on the presynaptic neuron.
    '''
    data = np.repeat(np.arange(N), N)
    delays = DynamicArray1D(data.shape, dtype=np.int32)
    delays[:] = data
    synapses = [DynamicArray1D(N, dtype=np.int32) for _ in xrange(N)]
    for i in xrange(N):
        synapses[i][:] = np.arange(N) + i*N
    return synapses, delays


def create_one_to_one(N):
    '''
    Return a tuple containing `synapses` and `delays` in the form that is needed
    for the `SpikeQueue` initializer.
    Every synapse has a delay depending on the presynaptic neuron.
    '''
    data = np.arange(N)
    delays = DynamicArray1D(data.shape, dtype=np.int32)
    delays[:] = data
    data = np.arange(N)
    synapses = [DynamicArray1D(1, dtype=np.int32) for _ in xrange(N)]
    for i in xrange(N):
        synapses[i][:] = i
    return synapses, delays


def test_spikequeue():
    N = 100
    synapses, delays = create_one_to_one(N)
    queue = SpikeQueue()
    queue.compress(delays, synapses, N)
    queue.push(np.arange(N, dtype=np.int32), delays)
    for i in xrange(N):
        assert_equal(queue.peek(), np.array([i]))
        queue.next()
    for i in xrange(N):
        assert_equal(queue.peek(), np.array([]))
        queue.next()

    synapses, delays = create_all_to_all(N)

    queue = SpikeQueue()
    queue.compress(delays, synapses, N*N)
    queue.push(np.arange(N*N, dtype=np.int32), delays)
    for i in xrange(N):
        assert_equal(queue.peek(), i*N + np.arange(N))
        queue.next()
    for i in xrange(N):
        assert_equal(queue.peek(), np.array([]))
        queue.next()


if __name__ == '__main__':
    test_spikequeue()
