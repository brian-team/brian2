"""
Brette R (2013). Sharpness of spike initiation in neurons explained by compartmentalization.
PLoS Comp Biol, doi: 10.1371/journal.pcbi.1003338.

Fig. 3. A, B. Kink with only Nav1.6 channels
"""
from brian2 import *
from params import *

#brian_prefs.codegen.target = 'weave'

defaultclock.dt=0.025*ms

# Morphology
morpho=Soma(50*um) # chosen for a target Rm
morpho.axon=Cylinder(diameter=1*um,length=300*um,n=300)

location=40*um # where Na channels are placed

# Channels
eqs='''
Im=gL*(EL-v)+gNa*m*(ENa-v) : amp/meter**2
dm/dt=(minf-m)/taum : 1 # simplified Na channel
minf=1/(1+exp((va-v)/ka)) : 1
gNa : siemens/meter**2
gL : siemens/meter**2
Iin : amp (point current)
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=Cm, Ri=Ri, method="exponential_euler")
compartment=morpho.axon[location].indices()[0]
neuron.v=EL
neuron.gNa[compartment]=gNa/neuron.area[compartment]
M=StateMonitor(neuron,('v','m'),record=True)

run(20*ms,report='text')
neuron.Iin[0]=gL*20*mV*neuron.area[0]
run(80*ms,report='text')

subplot(121)
plot(M.t/ms,M.v[0]/mV,'r')
plot(M.t/ms,M.v[compartment]/mV,'k')
plot(M.t/ms,M.m[compartment]*(80+60)-80,'k--') # open channels
ylim(-80,60)
xlabel('Time (ms)')
ylabel('V (mV)')
title('Voltage traces')

subplot(122)
dm=diff(M.v[0])/defaultclock.dt
dm40=diff(M.v[compartment])/defaultclock.dt
plot((M.v[0]/mV)[1:],dm,'r')
plot((M.v[compartment]/mV)[1:],dm40,'k')
xlim(-80,40)
xlabel('V (mV)')
ylabel('dV/dt (V/s)')
title('Phase plot')

show()
