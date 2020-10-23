#!/usr/bin/env python
"""
Cochlear neuron model of Rothman & Manis
----------------------------------------
Rothman JS, Manis PB (2003) The roles potassium currents play in
regulating the electrical activity of ventral cochlear nucleus neurons.
J Neurophysiol 89:3097-113.

All model types differ only by the maximal conductances.

Adapted from their Neuron implementation by Romain Brette
"""
from brian2 import *

#defaultclock.dt=0.025*ms # for better precision

'''
Simulation parameters: choose current amplitude and neuron type
(from type1c, type1t, type12, type 21, type2, type2o)
'''
neuron_type = 'type1c'
Ipulse = 250*pA

C = 12*pF
Eh = -43*mV
EK = -70*mV  # -77*mV in mod file
El = -65*mV
ENa = 50*mV
nf = 0.85  # proportion of n vs p kinetics
zss = 0.5  # steady state inactivation of glt
temp = 22.  # temperature in degree celcius
q10 = 3. ** ((temp - 22) / 10.)
# hcno current (octopus cell)
frac = 0.0
qt = 4.5 ** ((temp - 33.) / 10.)

# Maximal conductances of different cell types in nS
maximal_conductances = dict(
type1c=(1000, 150, 0, 0, 0.5, 0, 2),
type1t=(1000, 80, 0, 65, 0.5, 0, 2),
type12=(1000, 150, 20, 0, 2, 0, 2),
type21=(1000, 150, 35, 0, 3.5, 0, 2),
type2=(1000, 150, 200, 0, 20, 0, 2),
type2o=(1000, 150, 600, 0, 0, 40, 2) # octopus cell
)
gnabar, gkhtbar, gkltbar, gkabar, ghbar, gbarno, gl = [x * nS for x in maximal_conductances[neuron_type]]

gating_var = Equations('''d{name}/dt = q10*({name}__inf - {name})/tau_{name} : 1
                          {name}__inf = {rate_expression}                    : 1
                          tau_{name} = {tau_scale}/({forward_rate} + 
                                                    {reverse_rate})
                                       + {tau_base}                          : second''')

pos_sigmoid = Expression('1./(1+exp(-({voltage} - {midpoint}) / {scale}))')
sqrt_sigmoid = Expression('1./(1+exp(-({voltage} - {midpoint}) / {scale}))**0.5')
neg_sigmoid = Expression('1./(1+exp(({voltage} - {midpoint}) / {scale}))')
exp_voltage_dep = Expression('{magnitude}*exp(({voltage}-{midpoint})/{scale})')
neg_exp_voltage_dep = Expression('{magnitude}*exp(-({voltage}-{midpoint})/{scale})')

# Classical Na channel
ina = Equations('ina = gnabar*{m}**3*{h}*(ENa-v) : amp',
                m=gating_var(name='m',
                             rate_expression=pos_sigmoid(midpoint=-38., scale=7.),
                             forward_rate=exp_voltage_dep(magnitude=5., midpoint=-60, scale=18.),
                             reverse_rate=neg_exp_voltage_dep(magnitude=36., midpoint=-60, scale=25.),
                             tau_base=0.04*ms, tau_scale=10*ms),
                h=gating_var(name='h',
                             rate_expression=neg_sigmoid(midpoint=-65., scale=6.),
                             forward_rate=exp_voltage_dep(magnitude=7., midpoint=-60., scale=11.),
                             reverse_rate=neg_exp_voltage_dep(magnitude=10., midpoint=-60., scale=25.),
                             tau_base=0.6*ms, tau_scale=100*ms))

# KHT channel (delayed-rectifier K+)
ikht = Equations('ikht = gkhtbar*(nf*{n}**2 + (1-nf)*{p})*(EK-v) : amp',
                 n=gating_var(name='n',
                              rate_expression=sqrt_sigmoid(midpoint=-15, scale=5.),
                              forward_rate=exp_voltage_dep(magnitude=11., midpoint=-60, scale=24.),
                              reverse_rate=neg_exp_voltage_dep(magnitude=21., midpoint=-60, scale=23.),
                              tau_base=0.7*ms, tau_scale=100*ms),
                 p=gating_var(name='p',
                              rate_expression=pos_sigmoid(midpoint=-23., scale=6.),
                              forward_rate=exp_voltage_dep(magnitude=4., midpoint=-60., scale=32.),
                              reverse_rate=neg_exp_voltage_dep(magnitude=5., midpoint=-60., scale=22.),
                              tau_base=5*ms, tau_scale=100*ms))

# Ih channel (subthreshold adaptive, non-inactivating)
eqs_ih = """
ih = ghbar*r*(Eh-v) : amp
dr/dt=q10*(rinf-r)/rtau : 1
rinf = 1. / (1+exp((vu + 76.) / 7.)) : 1
rtau = ((100000. / (237.*exp((vu+60.) / 12.) + 17.*exp(-(vu+60.) / 14.))) + 25.)*ms : second
"""

# KLT channel (low threshold K+)
eqs_klt = """
iklt = gkltbar*w**4*z*(EK-v) : amp
dw/dt=q10*(winf-w)/wtau : 1
dz/dt=q10*(zinf-z)/ztau : 1
winf = (1. / (1 + exp(-(vu + 48.) / 6.)))**0.25 : 1
zinf = zss + ((1.-zss) / (1 + exp((vu + 71.) / 10.))) : 1
wtau = ((100. / (6.*exp((vu+60.) / 6.) + 16.*exp(-(vu+60.) / 45.))) + 1.5)*ms : second
ztau = ((1000. / (exp((vu+60.) / 20.) + exp(-(vu+60.) / 8.))) + 50)*ms : second
"""

# Ka channel (transient K+)
eqs_ka = """
ika = gkabar*a**4*b*c*(EK-v): amp
da/dt=q10*(ainf-a)/atau : 1
db/dt=q10*(binf-b)/btau : 1
dc/dt=q10*(cinf-c)/ctau : 1
ainf = (1. / (1 + exp(-(vu + 31) / 6.)))**0.25 : 1
binf = 1. / (1 + exp((vu + 66) / 7.))**0.5 : 1
cinf = 1. / (1 + exp((vu + 66) / 7.))**0.5 : 1
atau =  ((100. / (7*exp((vu+60) / 14.) + 29*exp(-(vu+60) / 24.))) + 0.1)*ms : second
btau =  ((1000. / (14*exp((vu+60) / 27.) + 29*exp(-(vu+60) / 24.))) + 1)*ms : second
ctau = ((90. / (1 + exp((-66-vu) / 17.))) + 10)*ms : second
"""

# Leak
eqs_leak = """
ileak = gl*(El-v) : amp
"""

# h current for octopus cells
eqs_hcno = """
ihcno = gbarno*(h1*frac + h2*(1-frac))*(Eh-v) : amp
dh1/dt=(hinfno-h1)/tau1 : 1
dh2/dt=(hinfno-h2)/tau2 : 1
hinfno = 1./(1+exp((vu+66.)/7.)) : 1
tau1 = bet1/(qt*0.008*(1+alp1))*ms : second
tau2 = bet2/(qt*0.0029*(1+alp2))*ms : second
alp1 = exp(1e-3*3*(vu+50)*9.648e4/(8.315*(273.16+temp))) : 1
bet1 = exp(1e-3*3*0.3*(vu+50)*9.648e4/(8.315*(273.16+temp))) : 1 
alp2 = exp(1e-3*3*(vu+84)*9.648e4/(8.315*(273.16+temp))) : 1
bet2 = exp(1e-3*3*0.6*(vu+84)*9.648e4/(8.315*(273.16+temp))) : 1
"""

eqs =Equations("""
dv/dt = (ileak + {currents} + iklt + ika + ih + ihcno + I)/C : volt
vu = v/mV : 1  # unitless v
I : amp
""")
eqs = eqs(currents=[ina, ikht], voltage='vu')
eqs += Equations(eqs_leak) + Equations(eqs_ka) + Equations(eqs_ih) + Equations(eqs_klt) + Equations(eqs_hcno)


neuron = NeuronGroup(1, eqs, method='exponential_euler')
neuron.v = El

run(50*ms, report='text')  # Go to rest

M = StateMonitor(neuron, 'v', record=0)
neuron.I = Ipulse

run(100*ms, report='text')

plot(M.t / ms, M[0].v / mV)
xlabel('t (ms)')
ylabel('v (mV)')
show()
