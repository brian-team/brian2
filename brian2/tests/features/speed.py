'''
Check the speed of different Brian 2 configurations
'''
from brian2 import *
from brian2.tests.features import SpeedTest

class LinearNeuronsOnly(SpeedTest):
    
    category = "Neurons only"
    name = "Linear neurons only"
    tags = ["Neurons", "Linear"]
    n_range = [10, 100, 1000, 10000, 100000, 1000000]
    n_label = 'Num neurons'
    
    def run(self):
        self.tau = tau = 1*second
        self.v_init = linspace(0.1, 1, self.n)
        self.duration = 100*ms
        G = self.G = NeuronGroup(self.n, 'dv/dt=-v/tau:1')
        self.G.v = self.v_init
        run(self.duration)

