"""
Brette R (2013). Sharpness of spike initiation in neurons explained by compartmentalization.
PLoS Comp Biol, doi: 10.1371/journal.pcbi.1003338.

Fig. 3. A, B. Kink with only Nav1.6 channels
"""
from brian2 import *
from params import *

codegen.target='numpy'

defaultclock.dt = 0.025*ms

# Morphology
morpho = Soma(50*um)  # chosen for a target Rm
morpho.axon = Cylinder(diameter=1*um, length=300*um, n=300)

location = 40*um  # where Na channels are placed

# Channels
eqs='''
Im = gL*(EL - v) + gNa*m*(ENa - v) : amp/meter**2
dm/dt = (minf - m) / taum : 1 # simplified Na channel
minf = 1 / (1 + exp((va - v) / ka)) : 1
gNa : siemens/meter**2
Iin : amp (point current)
'''

neuron = SpatialNeuron(morphology=morpho, model=eqs, Cm=Cm, Ri=Ri,
                       method="exponential_euler")

compartment = morpho.axon[location]
neuron.v = EL
neuron.gNa[compartment] = gNa_0/neuron.area[compartment]
M = StateMonitor(neuron, ['v', 'm'], record=True)

run(20*ms, report='text')
neuron.Iin[0] = gL * 20*mV * neuron.area[0]
run(80*ms, report='text')

subplot(121)
plot(M.t/ms, M[0].v/mV, 'r')
plot(M.t/ms, M[compartment].v/mV, 'k')
plot(M.t/ms, M[compartment].m*(80+60)-80, 'k--')  # open channels
ylim(-80, 60)
xlabel('Time (ms)')
ylabel('V (mV)')
title('Voltage traces')

subplot(122)
dm = diff(M[0].v) / defaultclock.dt
dm40 = diff(M[compartment].v) / defaultclock.dt
plot((M[0].v/mV)[1:], dm/(volt/second), 'r')
plot((M[compartment].v/mV)[1:], dm40/(volt/second), 'k')
xlim(-80, 40)
xlabel('V (mV)')
ylabel('dV/dt (V/s)')
title('Phase plot')

show()
