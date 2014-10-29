'''
Check that the basic features of `NeuronGroup` are available and correct.
'''
from brian2 import *
from brian2.tests.features import FeatureTest, InaccuracyError

class NeuronGroupIntegrationLinear(FeatureTest):
    
    category = "Basic NeuronGroup features"
    name = "Linear integration"
    tags = ["NeuronGroup", "Network", "Network.run"]
    
    def run(self):
        self.tau = tau = 1*second
        self.v_init = linspace(0.1, 1, 10)
        self.duration = 100*ms
        self.G = NeuronGroup(10, 'dv/dt=-v/tau:1')
        self.G.v = self.v_init
        self.net = Network(self.G)
        self.net.run(self.duration)
        
    def results(self):
        return self.G.v[:]
            
    def check(self, maxrelerr, v_end):
        v_correct = self.v_init*exp(-self.duration/self.tau)
        err = amax(abs(v_end-v_correct)/v_correct)
        if err>maxrelerr:
            raise InaccuracyError(err)
            
    
class NeuronGroupIntegrationEuler(FeatureTest):
    
    category = "Basic NeuronGroup features"
    name = "Euler integration"
    tags = ["NeuronGroup", "Network", "Network.run"]
    
    def run(self):
        self.tau = tau = 1*second
        self.v_init = linspace(0.1, 1, 10)
        self.duration = 100*ms
        self.G = NeuronGroup(10, 'dv/dt=-v**1.1/tau:1')
        self.G.v = self.v_init
        self.net = Network(self.G)
        self.net.run(self.duration)
        
    def results(self):
        return self.G.v[:]
    
    def compare(self, maxrelerr, v_base, v_test):
        err = amax(abs(v_base-v_test)/v_base)
        if err>maxrelerr:
            raise InaccuracyError(err)

        
if __name__=='__main__':
#    ft = NeuronGroupIntegrationLinear()
#    ft.run()
#    res = ft.results()
#    ft.check(res)
    
    ft = NeuronGroupIntegrationEuler()
    ft.run()
    res = ft.results()
    