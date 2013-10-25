import tempfile
import os

from nose import with_setup
import numpy

from brian2 import *
from brian2.devices.cpp_standalone import *


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
    
    net = Network(G,
                  M,
                  S,
                  )
    net.run(100*ms)
    tempdir = tempfile.mkdtemp()
    build(project_dir=tempdir, compile_project=True, run_project=True,
          with_output=with_output)
    i = numpy.fromfile(os.path.join(tempdir, 'results', 'spikemonitor_codeobject_i'),
                       dtype=numpy.int32)
    t = numpy.fromfile(os.path.join(tempdir, 'results', 'spikemonitor_codeobject_t'),
                       dtype=numpy.float64)
    assert len(i)==17741
    assert len(t)==17741
    assert t[0] == 0.
    assert t[-1] == float(100*ms - defaultclock.dt)
    
if __name__=='__main__':
    # Print the debug output when testing this file only but not when running
    # via nose test
    test_cpp_standalone(with_output=True)
