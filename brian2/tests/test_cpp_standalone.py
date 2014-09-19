import tempfile
import os

from nose import with_setup
import numpy
from numpy.testing.utils import assert_allclose

from brian2 import *
from brian2.devices.cpp_standalone import cpp_standalone_device


def restore_device():
    cpp_standalone_device.reinit()
    set_device('runtime')
    restore_initial_state()


@with_setup(teardown=restore_device)
def test_cpp_standalone(with_output=False):
    set_device('cpp_standalone')
    ##### Define the model
    tau = 1*ms
    eqs = '''
    dV/dt = (-40*mV-V)/tau : volt (unless refractory)
    '''
    threshold = 'V>-50*mV'
    reset = 'V=-60*mV'
    refractory = 5*ms
    N = 1000
    
    G = NeuronGroup(N, eqs,
                    reset=reset,
                    threshold=threshold,
                    refractory=refractory,
                    name='gp')
    G.V = '-i*mV'
    M = SpikeMonitor(G)
    S = Synapses(G, G, 'w : volt', pre='V += w')
    S.connect('abs(i-j)<5 and i!=j')
    S.w = 0.5*mV
    S.delay = '0*ms'

    net = Network(G, M, S)
    net.run(100*ms)
    tempdir = tempfile.mkdtemp()
    if with_output:
        print tempdir
    device.build(project_dir=tempdir, compile_project=True, run_project=True,
                 with_output=with_output)
    # we do an approximate equality here because depending on minor details of how it was compiled, the results
    # may be slightly different (if -ffast-math is on)
    assert len(M.i)>=17000 and len(M.i)<=18000
    assert len(M.t) == len(M.i)
    assert M.t[0] == 0.
    assert M.t[-1] == 100*ms - defaultclock.dt

@with_setup(teardown=restore_device)
def test_multiple_connects(with_output=False):
    set_device('cpp_standalone')
    G = NeuronGroup(10, 'v:1')
    S = Synapses(G, G, 'w:1')
    S.connect([0], [0])
    S.connect([1], [1])
    tempdir = tempfile.mkdtemp()
    if with_output:
        print tempdir
    run(0*ms)
    device.build(project_dir=tempdir, compile_project=True, run_project=True,
                 with_output=True)
    assert len(S) == 2 and len(S.w[:]) == 2


@with_setup(teardown=restore_device)
def test_storing_loading(with_output=False):
    set_device('cpp_standalone')
    G = NeuronGroup(10, '''v : volt
                           x : 1
                           n : integer
                           b : boolean''')
    v = np.arange(10)*volt
    x = np.arange(10, 20)
    n = np.arange(20, 30)
    b = np.array([True, False]).repeat(5)
    G.v = v
    G.x = x
    G.n = n
    G.b = b
    S = Synapses(G, G, '''v : volt
                          x : 1
                          n : integer
                          b : boolean''', connect='i==j')
    S.v = v
    S.x = x
    S.n = n
    S.b = b
    run(0*ms)
    tempdir = tempfile.mkdtemp()
    if with_output:
        print tempdir
    device.build(project_dir=tempdir, compile_project=True, run_project=True,
                 with_output=True)
    assert_allclose(G.v[:], v)
    assert_allclose(S.v[:], v)
    assert_allclose(G.x[:], x)
    assert_allclose(S.x[:], x)
    assert_allclose(G.n[:], n)
    assert_allclose(S.n[:], n)
    assert_allclose(G.b[:], b)
    assert_allclose(S.b[:], b)


if __name__=='__main__':
    # Print the debug output when testing this file only but not when running
    # via nose test
    for t in [
             test_cpp_standalone,
             test_multiple_connects,
             test_storing_loading
             ]:
        t(with_output=True)
        restore_device()
