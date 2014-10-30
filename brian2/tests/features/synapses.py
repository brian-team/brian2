'''
Check that the features of `Synapses` are available and correct.
'''
from brian2 import *
from brian2.tests.features import FeatureTest, InaccuracyError

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
        S = Synapses(G, H, pre='V += 1', connect='i==j')
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
        S = Synapses(H, G, post='V_pre += 1', connect='i==j')
        self.H = H
        run(101*ms)
        
    def results(self):
        return self.H.V[:]
            
    compare = FeatureTest.compare_arrays

if __name__=='__main__':
    for ftc in [SynapsesPre, SynapsesPost]:
        ft = ftc()
        ft.run()
        print ft.results()
    