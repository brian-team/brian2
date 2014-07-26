'''
Hodgkin-Huxley equations (1952)
Takes about 6.5 s with weave for 1 biological second (just run time).

Conduction velocity is about 12.5 m/s (is it right?)

Additionally: we record spikes.
'''
from brian2 import *

brian_prefs.codegen.target = 'weave' # couldn't this be simpler?

defaultclock.dt = 0.05 * ms

morpho=Cylinder(length=10*cm, diameter=2*238*um, n=1000, type='axon')

El = 10.613* mV
ENa = 115*mV
EK = -12 * mV
gl = 0.3 * msiemens / cm ** 2
gNa0 = 120 * msiemens / cm ** 2
gK = 36 * msiemens / cm ** 2

# Typical equations
eqs=''' # The same equations for the whole neuron, but possibly different parameter values
Im=gl*(El-v)+gNa*m**3*h*(ENa-v)+gK*n**4*(EK-v)+I/area : amp/meter**2 # distributed transmembrane current
I:amp # applied current
dm/dt=alpham*(1-m)-betam*m : 1
dn/dt=alphan*(1-n)-betan*n : 1
dh/dt=alphah*(1-h)-betah*h : 1
alpham=(0.1/mV)*(-v+25*mV)/(exp((-v+25*mV)/(10*mV))-1)/ms : Hz
betam=4.*exp(-v/(18*mV))/ms : Hz
alphah=0.07*exp(-v/(20*mV))/ms : Hz
betah=1./(exp((-v+30*mV)/(10*mV))+1)/ms : Hz
alphan=(0.01/mV)*(-v+10*mV)/(exp((-v+10*mV)/(10*mV))-1)/ms : Hz
betan=0.125*exp(-v/(80*mV))/ms : Hz
gNa : siemens/meter**2
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs, threshold = "m > 0.5",
                       threshold_location = morpho.compartment(5*cm), refractory = "m > 0.4",
                       Cm=1 * uF / cm ** 2, Ri=35.4 * ohm * cm, method="exponential_euler")
neuron.v=0*mV
neuron.h=1
neuron.m=0
neuron.n=.5
neuron.I=0#*amp/cm**2
neuron.gNa=gNa0
#neuron[5*cm:10*cm].gNa=0*siemens/cm**2
M=StateMonitor(neuron,'v',record=True)
spikes=SpikeMonitor(neuron)

run(50*ms,report='text')
neuron.I[0]=1 * uA # current injection at one end
run(3*ms)
neuron.I=0*amp
run(50*ms,report='text')

subplot(211)
for i in range(10):
    plot(M.t/ms,M.v.T[:,i*100]/mV)
subplot(212)
plot(spikes.t/second, spikes.i, ',k')
show()
