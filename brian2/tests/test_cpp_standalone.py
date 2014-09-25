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


@with_setup(teardown=restore_device)
def test_openmp_consistency(with_output=False):

    set_device('cpp_standalone')

    n_cells    = 100
    n_recorded = 10
    numpy.random.seed(42)
    taum       = 20 * ms
    taus       = 5 * ms
    Vt         = -50 * mV
    Vr         = -60 * mV
    El         = -49 * mV
    fac        = (60 * 0.27 / 10)
    gmax       = 20*fac
    dApre      = .01
    taupre     = 20 * ms
    taupost    = taupre
    dApost     = -dApre * taupre / taupost * 1.05
    dApost    *=  0.1*gmax
    dApre     *=  0.1*gmax

    connectivity = numpy.random.randn(n_cells, n_cells)
    sources      = numpy.random.random_integers(0, n_cells-1, 10*n_cells)
    times        = 1*second*numpy.sort(numpy.random.rand(10*n_cells))
    v_init       = Vr + numpy.random.rand(n_cells) * (Vt - Vr)

    eqs  = Equations('''
    dv/dt = (g-(v-El))/taum : volt
    dg/dt = -g/taus         : volt
    ''')

    results = {}

    for n_threads in [0, 1, 2]:
        Synapses.__instances__().clear()
        device.reinit()
        brian_prefs.codegen.cpp_standalone.openmp_threads = n_threads                
        P    = NeuronGroup(n_cells, model=eqs, threshold='v>Vt', reset='v=Vr', refractory=5 * ms)
        Q    = SpikeGeneratorGroup(n_cells, sources, times)
        P.v  = v_init
        P.g  = 0 * mV
        S    = Synapses(P, P, 
                            model = '''dApre/dt=-Apre/taupre    : 1 (event-driven)    
                                       dApost/dt=-Apost/taupost : 1 (event-driven)
                                       w                        : 1''', 
                            pre = '''g     += w*mV
                                     Apre  += dApre
                                     w      = w + Apost''',
                            post = '''Apost += dApost
                                      w      = w + Apre''',
                            connect=True)
        
        S.w       = fac*connectivity.flatten()

        T         = Synapses(Q, P, model = "w : 1", pre="g += w*mV", connect='i==j')
        T.w       = 10*fac

        spike_mon = SpikeMonitor(P)
        rate_mon  = PopulationRateMonitor(P)
        state_mon = StateMonitor(S, 'w', record=range(n_recorded), when=Clock(dt=0.1*second))
        v_mon     = StateMonitor(P, 'v', record=range(n_recorded))

        run(0.2 * second, report='text')

        tempdir = tempfile.mkdtemp()
        if with_output:
            print tempdir
        device.build(project_dir=tempdir, compile_project=True, run_project=True,
                     with_output=with_output)

        results[n_threads]      = {}
        results[n_threads]['w'] = state_mon.w
        results[n_threads]['v'] = v_mon.v
        results[n_threads]['s'] = spike_mon.num_spikes
        results[n_threads]['r'] = rate_mon.rate[:]
        print results[n_threads]

    for n_threads in [1, 2]:
        assert (results[n_threads]['w'] == results[0]['w']).all()
        assert (results[n_threads]['v'] == results[0]['v']).all()
        assert (results[n_threads]['r'] == results[0]['r']).all()
        assert (results[n_threads]['s'] == results[0]['s'])


if __name__=='__main__':
    # Print the debug output when testing this file only but not when running
    # via nose test
    for t in [
             test_cpp_standalone,
             test_multiple_connects,
             test_openmp_consistency
             ]:
        t(with_output=True)
        restore_device()
