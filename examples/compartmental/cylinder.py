'''
A short cylinder with constant injection at one end.
'''
from brian2 import *

defaultclock.dt = 0.01*ms

# Morphology
diameter = 1*um
length = 300*um
Cm = 1*uF/cm**2
Ri = 150*ohm*cm
N = 200
morpho = Cylinder(diameter=diameter, length=length, n=N)

# Passive channels
gL = 1e-4*siemens/cm**2
EL = -70*mV
eqs = '''
Im = gL * (EL - v) : amp/meter**2
I : amp (point current)
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=Cm, Ri=Ri,
                       method='exponential_euler')
neuron.v = EL

la = neuron.space_constant[0]
print("Electrotonic length: %s" % la)

neuron.I[0] = 0.02*nA # injecting at the left end
run(100*ms, report='text')

plot(neuron.distance/um, neuron.v/mV, 'kx')
# Theory
x = neuron.distance
ra = la * 4 * Ri / (pi * diameter**2)
theory = EL + ra * neuron.I[0] * cosh((length - x) / la) / sinh(length / la)
plot(x/um, theory/mV, 'r')
xlabel('x (um)')
ylabel('v (mV)')
show()
