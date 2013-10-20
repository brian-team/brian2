from numpy import *
from brian2 import *
from brian2.devices.cpp_standalone import *
from numpy.testing import assert_raises, assert_equal
from nose import with_setup
import tempfile
import os

@with_setup(teardown=restore_initial_state)
def test_cpp_standalone():
    previous_device = get_device()
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
    build(project_dir=tempdir, compile_project=True, run_project=True)
    S = loadtxt(os.path.join(tempdir, 'results', 'spikemonitor_codeobject.txt'), delimiter=',',
                dtype=[('i', int), ('t', float)])
    i = S['i']
    t = S['t']*second
    assert len(i)==17741

    # reset the device
    set_device(previous_device)
    
if __name__=='__main__':
    test_cpp_standalone()
    