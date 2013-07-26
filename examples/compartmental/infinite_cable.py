'''
An infinite cable.
'''
from pylab import *
from brian2 import *

brian_prefs.codegen.target = 'weave' # couldn't this be simpler?
#BrianLogger.log_level_debug()

# Morphology
morpho=Cylinder(diameter=1*um,length=3*mm,n=500)

# Passive channels
gL=1e-4*siemens/cm**2
EL=-70*mV
eqs='''
Im=gL*(EL-v)+I : amp/meter**2
dw/dt=-w/(20*ms) : 1
I : amp/meter**2
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1 * uF / cm ** 2, Ri=100 * ohm * cm)
neuron.v=EL
neuron.w=1
neuron.I=0*amp/cm**2

# Monitors
mon=StateMonitor(neuron,('v','w'),record=range(0,len(neuron)/2,20))
#run(1000*ms,report='text')
#exit()

run(1*ms)
neuron.I[len(neuron)/2]=0.2*nA/neuron.area[len(neuron)/2] # injecting in the middle
run(1*ms)
neuron.I=0*amp/meter**2
run(50*ms,report='text')

subplot(211)
plot(mon.t/ms,mon.v/mV)
subplot(212)
plot(mon.t/ms,mon.w)
show()
