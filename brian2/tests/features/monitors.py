'''
Check that various monitors work correctly.
'''

from brian2 import *
from brian2.tests.features import FeatureTest, InaccuracyError


class SpikeMonitorTest(FeatureTest):
    
    category = "Monitors"
    name = "SpikeMonitor"
    tags = ["NeuronGroup", "run",
            "SpikeMonitor"]
    
    def run(self):
        N = 100
        tau = 10*ms
        eqs = '''
        dv/dt = (I-v)/tau : 1
        I : 1
        '''
        self.G = G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0')
        G.I = linspace(0, 2, N)
        self.M = M = SpikeMonitor(G)
        run(100*ms)
        
    def results(self):
        return {'i': self.M.i[:], 't': self.M.t[:]}
            
    compare = FeatureTest.compare_arrays


class StateMonitorTest(FeatureTest):
    
    category = "Monitors"
    name = "StateMonitor"
    tags = ["NeuronGroup", "run",
            "StateMonitor"]
    
    def run(self):
        N = 10
        tau = 10*ms
        eqs = '''
        dv/dt = (I-v)/tau : 1
        I : 1
        '''
        self.G = G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0.1')
        G.v = 0.1
        G.I = linspace(1.1, 2, N)
        self.M = M = StateMonitor(G, 'v', record=True)
        run(100*ms)
        
    def results(self):
        return self.M.v[:]
            
    compare = FeatureTest.compare_arrays
