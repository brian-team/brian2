'''
An infinite cable.
'''
from brian2 import *

# Morphology
morpho=Cylinder(diameter=1*um,length=3*mm,n=500)

# Passive channels
gL=1e-4*siemens/cm**2
EL=-70*mV
eqs='''
Im=gL*(EL-v)+I : amp/cm**2
I : amp/cm**2
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1 * uF / cm ** 2, Ri=100 * ohm * cm)
neuron.v=EL
neuron.I=0*amp/cm**2

# Monitors
mon=StateMonitor(neuron,'v',record=True)

run(1*ms)
neuron.I[len(neuron)/2]=0.2*nA/neuron.area[len(neuron)/2] # injecting in the middle
run(1*ms)
neuron.I=0*amp
run(50*ms,report='text')

for i in range(0,len(neuron)/2,20):
    plot(mon.times/ms,mon[i]/mV)
show()
