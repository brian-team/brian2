'''
Input-Frequency curve of a HH model
Network: 100 unconnected Hodgin-Huxley neurons with an input current I.
The input is set differently for each neuron.

This simulation should use exponential Euler integration
'''
from brian2 import *

BrianLogger.log_level_info()

#brian_prefs.codegen.target = 'weave'

N = 100

# Parameters
area = 20000 * umetre ** 2
Cm = (1 * ufarad * cm ** -2) * area
gl = (5e-5 * siemens * cm ** -2) * area
El = -65 * mV
EK = -90 * mV
ENa = 50 * mV
g_na = (100 * msiemens * cm ** -2) * area
g_kd = (30 * msiemens * cm ** -2) * area
VT = -63 * mV

# The model
eqs = Equations('''
dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
    (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
    (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
    (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
I : amp
''')
# Threshold and refractoriness are only used for spike counting
group = NeuronGroup(N, model=eqs, threshold='not_refractory and (v > -40*mV)',
                    refractory='v > -40*mV')
group.v = El
group.I = linspace(0 * nA, 0.7 * nA, N)

monitor = SpikeMonitor(group)

duration = 2 * second
run(duration)
plot(group.I / nA, monitor.count / duration)
show()
