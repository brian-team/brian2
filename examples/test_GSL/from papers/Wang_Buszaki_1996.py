#!/usr/bin/env python
"""
August 1st, 2017: adapted to run with GSL by Charlee Fletterman.
Tried with: weave, cython and cpp_standalone. Figure looks the same as conventional brian (by eye)
=============================
Wang-Buszaki model
------------------

J Neurosci. 1996 Oct 15;16(20):6402-13.
Gamma oscillation by synaptic inhibition in a hippocampal interneuronal network model.
Wang XJ, Buzsaki G.

Note that implicit integration (exponential Euler) cannot be used, and therefore
simulation is rather slow.
"""
from brian2 import *
#set_device('cpp_standalone') # comment out if you want to test runtime device
prefs.codegen.target = 'cython' # comment out together with previous line to test for weave

defaultclock.dt = 0.01*ms

Cm = 1*uF # /cm**2
Iapp = 2*uA
gL = 0.1*msiemens
EL = -65*mV
ENa = 55*mV
EK = -90*mV
gNa = 35*msiemens
gK = 9*msiemens

eqs = '''
dv/dt = (-gNa*m**3*h*(v-ENa)-gK*n**4*(v-EK)-gL*(v-EL)+Iapp)/Cm : volt
m = alpha_m/(alpha_m+beta_m) : 1
alpha_m = -0.1/mV*(v+35*mV)/(exp(-0.1/mV*(v+35*mV))-1)/ms : Hz
beta_m = 4*exp(-(v+60*mV)/(18*mV))/ms : Hz
dh/dt = 5*(alpha_h*(1-h)-beta_h*h) : 1
alpha_h = 0.07*exp(-(v+58*mV)/(20*mV))/ms : Hz
beta_h = 1./(exp(-0.1/mV*(v+28*mV))+1)/ms : Hz
dn/dt = 5*(alpha_n*(1-n)-beta_n*n) : 1
alpha_n = -0.01/mV*(v+34*mV)/(exp(-0.1/mV*(v+34*mV))-1)/ms : Hz
beta_n = 0.125*exp(-(v+44*mV)/(80*mV))/ms : Hz
'''

neuron = NeuronGroup(1, eqs, method='GSL_stateupdater')
neuron.v = -70*mV
neuron.h = 1
M = StateMonitor(neuron, 'v', record=0)

run(100*ms, report='text')

plot(M.t/ms, M[0].v/mV)
show()
