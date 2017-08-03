#import matplotlib  # DELETE
#matplotlib.use('agg')  # DELETE

# comment on 9th of August 2017 by Charlee Fletterman: not that interesting for GSL because astrocyte equations use
# stochastic variables. Neuron and synapse equations not very complicated

from brian2 import *

GSL=True
if GSL:
    neuron_method = 'GSL_stateupdater'
    synapse_method = 'GSL_stateupdater'
else:
    neuron_method = 'euler'
    synapse_method = 'linear'

# set_device('cpp_standalone', directory='gchi_astrocyte')  # uncomment for fast simulation
seed(72930)  # to get identical figures for repeated runs

################################################################################
# Model parameters
################################################################################
### General parameters
duration = 30*second    # Total simulation time
dt = 1*ms               # integrator/sampling step

### Neuron parameters
f_0 = 0.5*Hz             # Spike rate of the "source" neurons

### Synapse parameters
rho_c = 0.001            # synaptic vesicle-to-extracellular space volume ratio
Y_T = 500*mmole          # Total neurotransmitter synaptic resource (in terms of vesicular concentration)
Omega_c = 40/second      # Neurotransmitter clearance rate

### Astrocyte parameters
# ---  Calcium fluxes
O_P = 0.9*umole/second   # Maximal Ca^2+ uptake rate
K_P = 0.1 * umole        # Ca2+ affinity of SERCAs
C_T = 2*umole            # Total ER Ca^2+ content
rho_A = 0.18             # ER-to-cytoplasm volume ratio
Omega_C = 6/second       # Maximal Ca^2+ release rate by IP_3Rs
Omega_L = 0.1/second     # Maximal Ca^2+ leak rate,
# --- IP_3R kinectics
d_1 = 0.13*umole         # IP_3 binding affinity
d_2 = 1.05*umole         # Inactivating Ca^2+ binding affinity
O_2 = 0.2/umole/second   # Inactivating Ca^2+ binding rate
d_3 = 0.9434*umole       # IP_3 binding affinity (with Ca^2+ inactivation)
d_5 = 0.08*umole         # Activating Ca^2+ binding affinity
# --- Agonist-dependent IP_3 production
O_beta = 5*umole/second  # Maximal rate of IP_3 production by PLCbeta
O_N = 0.3/umole/second   # Agonist binding rate
Omega_N = 0.5/second     # Inactivation rate of GPCR signalling
K_KC = 0.5*umole         # Ca^2+ affinity of PKC
zeta = 10                # Maximal reduction of receptor affinity by PKC
# --- IP_3 production
O_delta = 0.2 *umole/second # Maximal rate of IP_3 production by PLCdelta
kappa_delta = 1.5 * umole # Antagonizing IP_3 affinity
K_delta = 0.3*umole       # Ca^2+ affinity of PLCdelta
# --- IP_3 degradation
Omega_5P = 0.1/second    # Maximal rate of IP_3 degradation by IP-5P
K_D = 0.5*umole          # Ca^2+ affinity of IP3-3K
K_3K = 1*umole           # IP_3 affinity of IP_3-3K
O_3K = 4.5*umole/second  # Maximal rate of IP_3 degradation by IP_3-3K
# --- IP_3 diffusion
F = 2*umole/second       # GJC IP_3 permeability (nonlinear)
I_Theta = 0.3*umole      # Threshold IP_3 gradient for diffusion
omega_I = 0.05*umole     # Scaling factor of diffusion

################################################################################
# Model definition
################################################################################
defaultclock.dt = dt  # Set the integration time

### "Neurons"
# (We are only interested in the activity of the synapse, so we replace the
# neurons by trivial "dummy" groups
# # Regular spiking neuron
source_neurons = NeuronGroup(1, 'dx/dt = f_0 : 1', threshold='x>1', reset='x=0',
                             method=neuron_method)
# source_neurons = PoissonGroup(1, rates=f_0)
target_neurons = NeuronGroup(1, '')


### Synapses
# Our synapse model is trivial, we are only interested in its neurotransmitter
# release
# INCLUDE BEGIN
synapses_eqs = 'dY_S/dt = -Omega_c * Y_S : mole (clock-driven)'
synapses_action = 'Y_S += rho_c * Y_T'
synapses = Synapses(source_neurons, target_neurons,
                    model=synapses_eqs, on_pre=synapses_action,
                    method=synapse_method)
synapses.connect()
# INCLUDE END

### Astrocytes
# We are modelling two astrocytes exhibiting different patterns of Ca^2+ oscillations
# INCLUDE BEGIN
astro_eqs = '''
# Fraction of activated astrocyte receptors:
dGamma_A/dt = O_N * Y_S * (1 - Gamma_A) -
              Omega_N*(1 + zeta * C/(C + K_KC)) * Gamma_A : 1

# IP_3 dynamics:
dI/dt = J_beta + J_delta - J_3K - J_5P : mole
J_beta = O_beta * Gamma_A : mole/second
J_delta = O_delta/(1 + I/kappa_delta) * C**2/(C**2 + K_delta**2) : mole/second
J_3K = O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K) : mole/second
J_5P = Omega_5P*I : mole/second

# Ca^2+-induced Ca^2+ release:
dC/dt = J_r + J_l - J_p : mole
# IP3R de-inactivation probability
dh/dt = (h_inf - clip(h,0,1))/tau_h*(1 + noise*xi*tau_h**0.5) : 1
J_r = (Omega_C * m_inf**3 * h**3) * (C_T - (1 + rho_A)*C) : mole/second
J_l = Omega_L * (C_T - (1 + rho_A)*C) : mole/second
J_p = O_P * C**2/(C**2 + K_P**2) : mole/second
m_inf = I/(I + d_1) * C/(C + d_5) : 1
h_inf = Q_2/(Q_2 + C) : 1
tau_h = 1/(O_2 * (Q_2 + C)) : second
Q_2 = d_2 * (I + d_1)/(I + d_3) : mole

# Neurotransmitter concentration in the extracellular space
Y_S : mole
# Noise flag
noise : 1 (constant)
'''
method = 'milstein' # Noise is multiplicative in our case
astrocytes = NeuronGroup(2, astro_eqs, method=method)
# INCLUDE END
astrocytes.h = 0.9 # IP3Rs are initially mostly available for CICR
# The first astrocyte is deterministic,
# while the other one displays stochastic Ca^2+ dynamics
# INCLUDE BEGIN
astrocytes.noise = [0, 1]
# INCLUDE END
# Connection between synapses and astrocytes (both astrocytes receive the
# same input from the synapse). Note that in this special case, where each
# astrocyte is only influenced by the neurotransmitter from a single synapse,
# the linked variable mechanism could be used instead. The mechanism used below
# is more general and can add the contribution of several synapses.
# INCLUDE BEGIN
syn_to_astro = Synapses(synapses, astrocytes,
                        'Y_S_post = Y_S_pre : mole (summed)')
syn_to_astro.connect()
# INCLUDE END
################################################################################
# Monitors
################################################################################
astro_mon = StateMonitor(astrocytes, variables=['Gamma_A', 'C', 'h', 'I'],
                         record=True)

################################################################################
# Simulation run
################################################################################
run(duration, report='text')

################################################################################
# Analysis and plotting
################################################################################
plt.style.use('figures.mplstyle')

# Plot Gamma_A
fig, ax = plt.subplots(4, 1)
ax[0].plot(astro_mon.t/second, astro_mon.Gamma_A.T)
ax[0].set(ylim=[0.0, 1.02], yticks=[0.0, 0.5, 1.0], xticks=[], ylabel=r'$\Gamma_{A}$')

# Plot I
ax[1].plot(astro_mon.t/second, astro_mon.I.T/umole)
ax[1].set(ylim=[0.0,2.5], yticks=arange(0.0, 3.1, 0.7), xticks=[], ylabel=r'$I$ ($\mu M$)')

# Plot C
ax[2].plot(astro_mon.t/second, astro_mon.C.T/umole)
ax[2].set(ylim=[0.0,1.2],yticks=arange(0.0, 1.6, 0.5), ylabel=r'$C$ ($\mu M$)',
          xticks=[])
ax[2].legend(['deterministic', 'stochastic'], loc='upper right')

# Plot h
ax[3].plot(astro_mon.t/second, astro_mon.h.T)
ax[3].set(xlabel='Time ($s$)', ylim=[0.0, 1.02], yticks=[0.0, 0.5, 1.0], ylabel='h')

# Save Figure for paper # DELETE
#plt.savefig('../text/figures/results/example_2_gchi_astrocyte_Figure.png')  # DELETE
plt.show()
