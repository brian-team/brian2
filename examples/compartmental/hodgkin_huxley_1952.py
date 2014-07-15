'''
Hodgkin-Huxley equations (1952)

Conduction velocity is about 12.5 m/s (is it right?)

update=5.1s
apply=3.6s - this is probably mostly nonlinear state updates
post_update=1.26 s

So there seems to be little space for optimization left (for this morphology)
Except perhaps using tables and such for exponential functions (or fastexp? could be in a #define).
'''
from pylab import *
from brian2 import *

brian_prefs.codegen.target = 'weave' # couldn't this be simpler?

morpho=Cylinder(length=10*cm, diameter=2*238*um, n=1000, type='axon')

El = 10.613* mV
ENa = 115*mV
EK = -12 * mV
gl = 0.3 * msiemens / cm ** 2
gNa = 120 * msiemens / cm ** 2
gK = 36 * msiemens / cm ** 2

# Typical equations
eqs=''' # The same equations for the whole neuron, but possibly different parameter values
Im=gl*(El-v)+gNa*m**3*h*(ENa-v)+gK*n**4*(EK-v)+I : amp/meter**2 # distributed transmembrane current
I:amp/meter**2 # applied current
dm/dt=alpham*(1-m)-betam*m : 1
dn/dt=alphan*(1-n)-betan*n : 1
dh/dt=alphah*(1-h)-betah*h : 1
alpham=(0.1/mV)*(-v+25*mV)/(exp((-v+25*mV)/(10*mV))-1)/ms : Hz
betam=4.*exp(-v/(18*mV))/ms : Hz
alphah=0.07*exp(-v/(20*mV))/ms : Hz
betah=1./(exp((-v+30*mV)/(10*mV))+1)/ms : Hz
alphan=(0.01/mV)*(-v+10*mV)/(exp((-v+10*mV)/(10*mV))-1)/ms : Hz
betan=0.125*exp(-v/(80*mV))/ms : Hz
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=1 * uF / cm ** 2, Ri=35.4 * ohm * cm, method="exponential_euler")
neuron.v=0*mV
neuron.h=1
neuron.m=0
neuron.n=.5
neuron.I=0*amp/cm**2
M=StateMonitor(neuron,'v',record=True)

#run(1*second,report='text')
#exit()

run(50*ms,report='text')
neuron.I[0]=1 * uA/neuron.area[0] # current injection at one end
run(3*ms)
neuron.I=0*amp/cm**2
run(50*ms,report='text')

for i in range(10):
    plot(M.t/ms,M.v.T[:,i*10]/mV) # this is really slow!
show()
