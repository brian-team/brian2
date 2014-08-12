import tempfile
import os

from nose import with_setup
import numpy

from brian2 import *
from brian2.devices.cpp_standalone import cpp_standalone_device


def restore_device():
    #Network.__instances__().clear()  #TODO
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
    device.build(project_dir=tempdir, compile_project=True, run_project=True,
                 with_output=True)
    assert len(S) == 2 and len(S.w[:]) == 2


if __name__=='__main__':
    # Print the debug output when testing this file only but not when running
    # via nose test
    for t in [
             test_cpp_standalone,
             test_multiple_connects
             ]:
        t(with_output=True)
        restore_device()
