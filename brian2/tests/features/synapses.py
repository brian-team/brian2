'''
Check that the features of `Synapses` are available and correct.
'''
from brian2 import *
from brian2.tests.features import FeatureTest, InaccuracyError
import numpy

class SynapsesPre(FeatureTest):
    
    category = "Synapses"
    name = "Presynaptic code"
    tags = ["NeuronGroup", "run",
            "Synapses", "Presynaptic code"]
    
    def run(self):
        tau = 5*ms
        eqs = '''
        dV/dt = k/tau : 1
        k : 1
        '''
        G = NeuronGroup(10, eqs, threshold='V>1', reset='V=0')
        G.k = linspace(1, 5, len(G))
        H = NeuronGroup(10, 'V:1')
        S = Synapses(G, H, on_pre='V += 1')
        S.connect(j='i')
        self.H = H
        run(101*ms)
        
    def results(self):
        return self.H.V[:]
            
    compare = FeatureTest.compare_arrays


class SynapsesPost(FeatureTest):
    
    category = "Synapses"
    name = "Postsynaptic code"
    tags = ["NeuronGroup", "run",
            "Synapses", "Postsynaptic code"]
    
    def run(self):
        tau = 5*ms
        eqs = '''
        dV/dt = k/tau : 1
        k : 1
        '''
        G = NeuronGroup(10, eqs, threshold='V>1', reset='V=0')
        G.k = linspace(1, 5, len(G))
        H = NeuronGroup(10, 'V:1')
        S = Synapses(H, G, on_post='V_pre += 1')
        S.connect(j='i')
        self.H = H
        run(101*ms)
        
    def results(self):
        return self.H.V[:]
            
    compare = FeatureTest.compare_arrays


class SynapsesSTDP(FeatureTest):
    
    category = "Synapses"
    name = "STDP"
    tags = ["NeuronGroup", "Threshold", "Reset", "Refractory",
            "run",
            "Synapses", "Postsynaptic code", "Presynaptic code",
            "SpikeMonitor", "StateMonitor",
            "SpikeGeneratorGroup",
            ]
    
    def run(self):
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
        # Only use one spike per time step (to rule out that a single source neuron
        # has more than one spike in a time step)
        times        = numpy.random.choice(numpy.arange(10*n_cells), 10*n_cells,
                                           replace=False)*ms
        v_init       = Vr + numpy.random.rand(n_cells) * (Vt - Vr)
    
        eqs  = Equations('''
        dv/dt = (g-(v-El))/taum : volt
        dg/dt = -g/taus         : volt
        ''')
        
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
                                      w      = w + Apre''')
        S.connect()
        
        S.w       = fac*connectivity.flatten()

        T         = Synapses(Q, P, model = "w : 1", on_pre="g += w*mV")
        T.connect(j='i')
        T.w       = 10*fac

        spike_mon = SpikeMonitor(P)
        state_mon = StateMonitor(S, 'w', record=range(n_recorded))
        v_mon     = StateMonitor(P, 'v', record=range(n_recorded))
        
        self.state_mon = state_mon
        self.spike_mon = spike_mon
        self.v_mon = v_mon

        run(0.2 * second, report='text')

    def results(self):
        return self.state_mon.w[:], self.v_mon.v[:], self.spike_mon.num_spikes

    def compare(self, maxrelerr, res1, res2):
        w1, v1, n1 = res1
        w2, v2, n2 = res2
        FeatureTest.compare_arrays(self, maxrelerr, w1, w2)
        FeatureTest.compare_arrays(self, maxrelerr, v1, v2)
        FeatureTest.compare_arrays(self, maxrelerr, array([n1], dtype=float),
                                   array([n2], dtype=float))


if __name__=='__main__':
    for ftc in [SynapsesPre, SynapsesPost]:
        ft = ftc()
        ft.run()
        print ft.results()
    