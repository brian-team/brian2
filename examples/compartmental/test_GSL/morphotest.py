from brian2 import *

# Morphology
morpho = Soma(30*um)
morpho.L = Cylinder(diameter=1*um, length=100*um, n=5)
morpho.LL = Cylinder(diameter=1*um, length=20*um, n=2)
morpho.R = Cylinder(diameter=1*um, length=100*um, n=5)

# Passive channels
gL = 1e-4*siemens/cm**2
EL = -70*mV
eqs = '''
Im = gL * (EL-v) : amp/meter**2
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs,
                       Cm=1*uF/cm**2, Ri=100*ohm*cm, method='exponential_euler')
neuron.v = arange(0, 13)*volt

print(neuron.v)
print(neuron.L.v)
print(neuron.LL.v)
print(neuron.L.main.v)
