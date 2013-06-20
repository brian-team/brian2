import timeit
import itertools

import numpy as np

GENERAL_SETUP =  ['import numpy as np',
                  'from brian2.tests.test_spikequeue import create_all_to_all, create_one_to_one',
                  'from brian2.units.stdunits import ms',
                  'from brian2.synapses.spikequeue import SpikeQueue']

def get_setup_code(N, create_func):
    return GENERAL_SETUP + [
        'synapses, delays = {}({})'.format(create_func, N),
        'queue = SpikeQueue(synapses, delays, 0.1*ms)']

def test_compress(N, create_func):
    setup_code = get_setup_code(N, create_func)
    number = 1000/N
    results = timeit.repeat('queue.compress()', ';'.join(setup_code), repeat=5,
                            number=number)
    return np.array(results) / number


def test_push(N, create_func):
    setup_code = get_setup_code(N, create_func) + ['queue.compress()']
    number = 5000/N
    results = timeit.repeat('queue.push(np.arange({}));queue.next()'.format(N),
                            ';'.join(setup_code), repeat=5,
                            number=number)
    return np.array(results) / number


def run_benchmark(test_func, N, create_func):
    result = test_func(N, create_func)
    print '{} -- {}({}) : {}'.format(test_func.__name__, create_func, N,
                                     np.median(result))


if __name__ == '__main__':
    for test, N, create_func in itertools.product((test_compress, test_push),
                                                  (10, 100),
                                                  ('create_all_to_all',
                                                   'create_one_to_one')):
        run_benchmark(test, N, create_func)
