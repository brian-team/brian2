#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
This is an implementation of a benchmark described
in the following review paper:

Simulation of networks of spiking neurons: A review of tools and strategies (2006).
Brette, Rudolph, Carnevale, Hines, Beeman, Bower, Diesmann, Goodman, Harris, Zirpe,
Natschläger, Pecevski, Ermentrout, Djurfeldt, Lansner, Rochel, Vibert, Alvarez, Muller,
Davison, El Boustani and Destexhe.
Journal of Computational Neuroscience

Benchmark 3: random network of HH neurons with exponential synaptic conductances

Clock-driven implementation
(no spike time interpolation)

R. Brette - Dec 2007
"""

from brian2 import *

defaultclock.dt = .1*ms

prefs.codegen.target = 'cython'
prefs.codegen.cpp.libraries += ['gsl', 'gslcblas']
prefs.codegen.cpp.headers += ['gsl/gsl_odeiv2.h']
prefs.codegen.cpp.include_dirs += ['/home/charlee/softwarefolder/gsl-2.3/gsl/']

# Parameters
area = 20000*umetre**2
Cm = (1*ufarad*cm**-2) * area
gl = (5e-5*siemens*cm**-2) * area

El = -60*mV
EK = -90*mV
ENa = 50*mV
g_na = (100*msiemens*cm**-2) * area
g_kd = (30*msiemens*cm**-2) * area
VT = -63*mV
# Time constants
taue = 5*ms
taui = 10*ms
# Reversal potentials
Ee = 0*mV
Ei = -80*mV
we = 6*nS  # excitatory synaptic weight
wi = 67*nS  # inhibitory synaptic weight

# The model
eqs = Equations('''
dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
         g_na*(m*m*m)*h*(v-ENa)-
         g_kd*(n*n*n*n)*(v-EK))/Cm : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
alpha_m = 0.32*(mV**-1)*(13*mV-v+VT)/
         (exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
beta_m = 0.28*(mV**-1)*(v-VT-40*mV)/
        (exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
alpha_n = 0.032*(mV**-1)*(15*mV-v+VT)/
         (exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
''')

## First run simulation for GSL
seed(0)

P_GSL = NeuronGroup(4000, model=eqs, threshold='v>-20*mV', refractory=3*ms,
                method='GSL_stateupdater')
P_GSL.state_updater.codeobj_class = GSLCythonCodeObject
Pe_GSL = P_GSL[:3200]
Pi_GSL = P_GSL[3200:]
Ce_GSL = Synapses(Pe_GSL, P_GSL, on_pre='ge+=we')
Ci_GSL = Synapses(Pi_GSL, P_GSL, on_pre='gi+=wi')
Ce_GSL.connect(p=0.02)
Ci_GSL.connect(p=0.02)

# Initialization
P_GSL.v = 'El + (randn() * 5 - 5)*mV'
P_GSL.ge = '(randn() * 1.5 + 4) * 10.*nS'
P_GSL.gi = '(randn() * 12 + 20) * 10.*nS'

# Record a few traces
trace_GSL = StateMonitor(P_GSL, 'v', record=[1, 10, 100])
network_GSL = Network(P_GSL, Ce_GSL, Ci_GSL, trace_GSL)
network_GSL.run(.1 * second, report='text')

## Then for exponential euler
seed(0)


P_brian = NeuronGroup(4000, model=eqs, threshold='v>-20*mV', refractory=3*ms,
                method='exponential_euler')
Pe_brian = P_brian[:3200]
Pi_brian = P_brian[3200:]
Ce_brian = Synapses(Pe_brian, P_brian, on_pre='ge+=we')
Ci_brian = Synapses(Pi_brian, P_brian, on_pre='gi+=wi')
Ce_brian.connect(p=0.02)
Ci_brian.connect(p=0.02)

# Initialization
P_brian.v = 'El + (randn() * 5 - 5)*mV'
P_brian.ge = '(randn() * 1.5 + 4) * 10.*nS'
P_brian.gi = '(randn() * 12 + 20) * 10.*nS'

# Record a few traces
trace_brian = StateMonitor(P_brian, 'v', record=[1, 10, 100])
network_brian = Network(P_brian, Ce_brian, Ci_brian, trace_brian)
network_brian.run(.1 * second, report='text')

plot(trace_GSL.t/ms, trace_GSL[1].v/mV)
plot(trace_GSL.t/ms, trace_GSL[10].v/mV)
plot(trace_GSL.t/ms, trace_GSL[100].v/mV)
plot(trace_brian.t/ms, trace_brian[1].v/mV, '--')
plot(trace_brian.t/ms, trace_brian[10].v/mV, '--')
plot(trace_brian.t/ms, trace_brian[100].v/mV, '--')
xlabel('t (ms)')
ylabel('v (mV)')
show()
