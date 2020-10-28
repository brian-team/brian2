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
power_sigmoid = Expression('1./(1+exp(-({voltage} - {midpoint}) / {scale}))**{power}')
neg_sigmoid = Expression('1./(1+exp(({voltage} - {midpoint}) / {scale}))')
neg_power_sigmoid = Expression('1./(1+exp(({voltage} - {midpoint}) / {scale}))**{power}')
shifted_neg_sigmoid = Expression('{shift} + (1.0-{shift})/(1+exp(({voltage} - {midpoint}) / {scale}))')
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
                              rate_expression=power_sigmoid(midpoint=-15, scale=5., power=0.5),
                              forward_rate=exp_voltage_dep(magnitude=11., midpoint=-60, scale=24.),
                              reverse_rate=neg_exp_voltage_dep(magnitude=21., midpoint=-60, scale=23.),
                              tau_base=0.7*ms, tau_scale=100*ms),
                 p=gating_var(name='p',
                              rate_expression=pos_sigmoid(midpoint=-23., scale=6.),
                              forward_rate=exp_voltage_dep(magnitude=4., midpoint=-60., scale=32.),
                              reverse_rate=neg_exp_voltage_dep(magnitude=5., midpoint=-60., scale=22.),
                              tau_base=5*ms, tau_scale=100*ms))

# Ih channel (subthreshold adaptive, non-inactivating)
ih = Equations('ih = ghbar*{r}*(Eh - v) : amp',
               r=gating_var(name='r',
                            rate_expression=neg_sigmoid(midpoint=-76., scale=7.),
                            forward_rate=exp_voltage_dep(magnitude=237., midpoint=-60., scale=12.),
                            reverse_rate=neg_exp_voltage_dep(magnitude=17, midpoint=-60, scale=14.),
                            tau_scale=100*second, tau_base=25*ms))

# KLT channel (low threshold K+)
iklt = Equations('iklt = gkltbar*{w}**4*{z}*(EK-v) : amp',
                 w=gating_var(name='w',
                              rate_expression=power_sigmoid(midpoint=-48., scale=6., power=0.25),
                              forward_rate=exp_voltage_dep(magnitude=6., midpoint=-60., scale=6.),
                              reverse_rate=neg_exp_voltage_dep(magnitude=16, midpoint=-60, scale=45.),
                              tau_scale=100*ms, tau_base=1.5*ms),
                 z=gating_var(name='z',
                              rate_expression=shifted_neg_sigmoid(shift=zss, midpoint=-71., scale=10.),
                              forward_rate=exp_voltage_dep(magnitude=1., midpoint=-60., scale=20.),
                              reverse_rate=neg_exp_voltage_dep(magnitude=1., midpoint=-60., scale=8.),
                              tau_scale=1*second, tau_base=50*ms))
# Ka channel (transient K+)
ika = Equations('ika = gkabar*{a}**4*{b}*{c}*(EK-v): amp',
                a=gating_var(name='a',
                             rate_expression=power_sigmoid(midpoint=-31., scale=6., power=0.25),
                             forward_rate=exp_voltage_dep(magnitude=7., midpoint=-60, scale=14.),
                             reverse_rate=neg_exp_voltage_dep(magnitude=29., midpoint=-60, scale=24.),
                             tau_scale=100*ms, tau_base=0.1*ms),
                b=gating_var(name='b',
                             rate_expression=neg_power_sigmoid(midpoint=-66., scale=7., power=0.5),
                             forward_rate=exp_voltage_dep(magnitude=14., midpoint=-60., scale=27.),
                             reverse_rate=neg_exp_voltage_dep(magnitude=29., midpoint=-60., scale=24.),
                             tau_scale=1*second, tau_base=1*ms),
                c=gating_var(name='c',
                             rate_expression=neg_power_sigmoid(midpoint=-66., scale=7., power=0.5),
                             forward_rate='1',
                             reverse_rate=neg_exp_voltage_dep(magnitude=1., midpoint=-66., scale=17.),
                             tau_scale=90*ms, tau_base=10*ms))
# Leak
ileak = Equations("ileak = gl*(El-v) : amp")

# h current for octopus cells (gating variables use different equations
gating_var_octopus = Equations('''d{name}/dt = (hinfno - {name})/tau_{name} : 1
                                  tau_{name} = beta_{name}/(qt*{tau_scale}*(1+alpha_{name}))*ms : second
                                  alpha_{name} = exp(1e-3*3*({voltage}-{midpoint})*9.648e4/(8.315*(273.16+temp))) : 1
                                  beta_{name} = exp(1e-3*3*0.3*({voltage}-{midpoint})*9.648e4/(8.315*(273.16+temp))) : 1 ''')

ihcno = Equations('''ihcno = gbarno*({h1}*frac + {h2}*(1-frac))*(Eh-v) : amp
                     hinfno = {inf_rate} : 1''',
                  inf_rate=neg_sigmoid(midpoint=-66., scale=7.),
                  h1=gating_var_octopus(name='h1', tau_scale=0.008, midpoint=-50.),
                  h2=gating_var_octopus(name='h2', tau_scale=0.0029, midpoint=-84.))

eqs =Equations("""dv/dt = ({currents} + I)/C : volt
                  vu = v/mV : 1  # unitless v
                  I : amp""",
               currents=[ileak, ina, ikht, ih, iklt, ika, ihcno], voltage='vu')

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
