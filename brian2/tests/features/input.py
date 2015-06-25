'''
Tests of input features
'''

from brian2 import *
from brian2.tests.features import FeatureTest, InaccuracyError

class SpikeGeneratorGroupTest(FeatureTest):
    
    category = "Input"
    name = "SpikeGeneratorGroup"
    tags = ["SpikeMonitor", "run",
            "SpikeGeneratorGroup"]
    
    def run(self):
        N = 10
        numspikes = 1000
        i = arange(numspikes) % N
        t = linspace(0, 1, numspikes)*(100*ms)
        G = SpikeGeneratorGroup(N, i, t)
        self.M = M = SpikeMonitor(G)
        run(100*ms)
        
    def results(self):
        return {'i': self.M.i[:], 't': self.M.t[:]}
            
    compare = FeatureTest.compare_arrays
