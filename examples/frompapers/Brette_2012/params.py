"""
Parameters for spike initiation simulations.
"""
from brian2.units import *

# Passive parameters
EL = -75*mV
S = 7.85e-9*meter**2  # area (sphere of 50 um diameter)
Cm = 0.75*uF/cm**2
gL = 1. / (30000*ohm*cm**2)
Ri = 150*ohm*cm

# Na channels
ENa = 60*mV
ka = 6*mV
va = -40*mV
gNa_0 = gL * 2*S
taum = 0.1*ms
