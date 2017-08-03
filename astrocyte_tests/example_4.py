"""
example_4.py

Show average synaptic release for a 3-step stimulus produced by
    - a standalone synapse (black)
    - a synapse + astrocyte in open-loop configuration (darkgreen)
    - a synapse + astrocyte in closed-loop configuraton (red)

"""
#import matplotlib  # DELETE
#matplotlib.use('agg')  # DELETE

# Comment on 9th of August, 2017 by Charlee Fletterman: doesn't run without adaptations (possibly because not latest version of master)

from brian2 import *

# set_device('cpp_standalone', directory='synapse_release')  # uncomment for fast simulation
seed(16283)  # to get identical figures for repeated runs

################################################################################
# Model parameters
################################################################################
### General parameters
N_neurons = 500
N_astro = 2
duration = 20*second
dt = 1*ms    # integrator/sampling step

### Neuron parameters

# ### Synapse parameters
### Synapse parameters
rho_c = 0.005            # synaptic vesicle-to-extracellular space volume ratio
Y_T = 500*mmole          # Total neurotransmitter synaptic resource (in terms of vesicular concentration)
Omega_c = 40/second      # Neurotransmitter clearance rate
U_0__star = 0.6          # Basal synaptic release probability
Omega_f = 3.33/second    # Facilitation rate
Omega_d = 2.0/second     # Depression rate
# --- Presynaptic receptors
O_G = 1.5/umole/second   # Agonist binding rate (activating)
Omega_G = 0.5/(60*second)# Agonist release rate (inactivating)

### Astrocyte parameters
# ---  Calcium fluxes
O_P = 0.9*umole/second   # Maximal Ca^2+ uptake rate
K_P = 0.05 * umole       # Ca2+ affinity of SERCAs
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
# --- IP_3 production
# --- Agonist-dependent IP_3 production
O_beta = 3.2*umole/second  # Maximal rate of IP_3 production by PLCbeta
O_N = 0.3/umole/second   # Agonist binding rate
Omega_N = 0.5/second     # Inactivation rate of GPCR signalling
K_KC = 0.5*umole         # Ca^2+ affinity of PKC
zeta = 10                # Maximal reduction of receptor affinity by PKC
# --- Endogenous IP3 production
O_delta = 0.6*umole/second # Maximal rate of IP_3 production by PLCdelta
kappa_delta = 1.5* umole  # Antagonizing IP_3 affinity
K_delta = 0.1*umole        # Ca^2+ affinity of PLCdelta
# --- IP_3 diffusion
F = 2*umole/second       # GJC IP_3 permeability (nonlinear)
I_Theta = 0.3*umole      # Threshold IP_3 gradient for diffusion
omega_I = 0.05*umole     # Scaling factor of diffusion
# --- IP_3 degradation
Omega_5P = 0.05/second    # Maximal rate of IP_3 degradation by IP-5P
K_D = 0.7*umole          # Ca^2+ affinity of IP3-3K
K_3K = 1.0*umole           # IP_3 affinity of IP_3-3K
O_3K = 4.5*umole/second  # Maximal rate of IP_3 degradation by IP_3-3K
# --- IP_3 diffusion
F = 2.0*umole/second       # GJC IP_3 permeability (nonlinear)
I_Theta = 0.3*umole      # Threshold IP_3 gradient for diffusion
omega_I = 0.05*umole     # Scaling factor of diffusion
# --- Gliotransmitter release and time course
C_Theta = 0.5*umole      # Ca^2+ threshold for exocytosis
Omega_A = 0.6/second     # Gliotransmitter recycling rate
U_A = 0.6                # Gliotransmitter release probability
G_T = 200*mmole          # Total vesicular gliotransmitter
rho_e = 6.5e-4           # Ratio of astrocytic vesicle volume/ESS volume
Omega_e = 60/second      # Gliotransmitter clearance rate
alpha = 0.0              # Gliotranmission type (assumed maximally inhibitory)

################################################################################
# Model definition
################################################################################
defaultclock.dt = dt  # Set the integration time

### "Neurons"
rate_in = TimedArray([0.011, 0.1, 1.1, 11] * Hz, dt=5*second)
source_neurons = PoissonGroup(N_neurons, rates='rate_in(t)')
target_neurons = NeuronGroup(N_neurons, '')

### Synapses
# Note that the synapse does not actually have any effect on the post-synaptic
# target
# Also note that for easier plotting we do not use the "event-driven" flag here,
# even though the value of u_S and x_S only needs to be updated on the arrival
# of a spike
synapses_eqs = '''
# Neurotransmitter
dY_S/dt = -Omega_c * Y_S : mole (clock-driven)
# Fraction of activated presynaptic receptors
dGamma_S/dt = O_G * G_A * (1 - Gamma_S) - Omega_G * Gamma_S : 1 (clock-driven)
# Usage of releasable neurotransmitter per single action potential:
du_S/dt = -Omega_f * u_S : 1 (event-driven)
# Fraction of synaptic neurotransmitter resources available for release:
dx_S/dt = Omega_d *(1 - x_S) : 1 (event-driven)
r_S : 1  # released synaptic neurotransmitter resources
G_A : mole  # gliotransmitter concentration in the extracellular space
'''
synapses_action = '''
U_0 = (1 - Gamma_S) * U_0__star + alpha * Gamma_S
u_S += U_0 * (1 - u_S)
r_S = u_S * x_S
x_S -= r_S
Y_S += rho_c * Y_T * r_S
'''
synapses = Synapses(source_neurons, target_neurons,
                    model=synapses_eqs, on_pre=synapses_action,
                    multisynaptic_index='k', method='linear')
# We create three synapses per connection: only the first two are modulated by
# the astrocyte however
synapses.connect('i==j', n=N_astro+1)
synapses.x_S = 1.0

### Astrocytes
# The astrocyte emits gliotransmitter when its Ca^2+ concentration crosses
# a threshold
astro_eqs = '''
# Fraction of activated astrocyte receptors:
dGamma_A/dt = O_N * Y_S * (1 - Gamma_A) - Omega_N*(1 + zeta * C/(C + K_KC)) * Gamma_A : 1

# IP_3 dynamics:
dI/dt = J_beta + J_delta - J_3K - J_5P + I_exogenous : mole
J_beta = O_beta * Gamma_A : mole/second
J_delta = O_delta/(1 + I/kappa_delta) * C**2/(C**2 + K_delta**2) : mole/second
J_3K = O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K) : mole/second
J_5P = Omega_5P*I : mole/second
# Exogenous stimulation
delta_I_bias = I - I_bias : mole
I_exogenous = -F/2*(1 + tanh((abs(delta_I_bias) -
                             I_Theta)/omega_I)) *
              sign(delta_I_bias) : mole/second
I_bias : mole (constant)

# Ca^2+-induced Ca^2+ release:
dC/dt = (Omega_C * m_inf**3 * h**3 + Omega_L) * (C_T - (1 + rho_A)*C) -
        O_P * C**2/(C**2 + K_P**2) : mole
dh/dt = (h_inf - h)/tau_h : 1  # IP3R de-inactivation probability
m_inf = I/(I + d_1) * C/(C + d_5) : 1
h_inf = Q_2/(Q_2 + C) : 1
tau_h = 1/(O_2 * (Q_2 + C)) : second
Q_2 = d_2 * (I + d_1)/(I + d_3) : mole

dx_A/dt = Omega_A * (1 - x_A) : 1  # Fraction of gliotransmitter resources available for release
dG_A/dt = -Omega_e*G_A : mole  # gliotransmitter concentration in the extracellular space

# Neurotransmitter concentration in the extracellular space
Y_S : mole
'''
glio_release = '''
G_A += rho_e * G_T * U_A * x_A
x_A -= U_A *  x_A
'''
astrocyte = NeuronGroup(N_astro*N_neurons, astro_eqs,
                        # The following formulation makes sure that a "spike" is
                        # only triggered at the first threshold crossing
                        threshold='C>C_Theta',
                        refractory='C>C_Theta',
                        # The gliotransmitter release happens when the threshold
                        # is crossed, in Brian terms it can therefore be
                        # considered a "reset"
                        reset=glio_release,
                        method='rk4')
astrocyte.h = 0.9
astrocyte.x_A = 1.0
# Only the second group of N_neurons astrocytes are activated by external stimulation
astrocyte.I_bias = (np.r_[np.zeros(N_neurons), np.ones(N_neurons)])*1.0*umole

## Connections
syn_to_astro = Synapses(synapses, astrocyte,
                        'Y_S_post = Y_S_pre : mole (summed)')
## Connect th first N_neurons synapses to the first N_neurons astrocytes,
syn_to_astro.connect(j='int(i / 3) for _ in range(1) if i % 3 == 0 ')

astro_to_syn = Synapses(astrocyte, synapses,
                        'G_A_post = G_A_pre : mole (summed)')
## Connect the first N_neurons astrocytes to the first N_neurons synapses (closed-loop configuration)
astro_to_syn.connect(j='i*3 for _ in range(1) if i < N_neurons')
## Connect the second astrocyte to the second group of N_neurons synapses (open-loop configuration).
astro_to_syn.connect(j='(i-N_neurons)*3 + 1 for _ in range(1) if i >= N_neurons and i < 2*N_neurons')


################################################################################
# Monitors
################################################################################
syn_mon = StateMonitor(synapses, 'Y_S', record=np.arange(N_neurons*(N_astro+1)),
                       dt=10*ms)

################################################################################
# Simulation run
################################################################################
run(duration, report='text')

################################################################################
# Analysis and plotting
################################################################################
plt.style.use('figures.mplstyle')

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(7,8), sharex=True,
                       gridspec_kw={'height_ratios': [1,3,3,3],
                                    'top': 0.98, 'bottom': 0.1,
                                    'left': 0.1, 'right': 0.95})

ax[0].plot(syn_mon.t/second, rate_in(syn_mon.t), '-', color='black')
ax[0].set(yscale='log', ylabel=r'$\nu_{in}$ (Hz)')

ax[1].plot(syn_mon.t / second, np.mean(syn_mon[synapses[:, :, 2]].Y_S / umole, axis=0),
           '-', color='black')
ax[1].set(ylabel=r'$\langle Y_S \rangle$ ($\mu$M)')
ax[1].legend(['no gliotransmission'], loc='upper left')

ax[2].plot(syn_mon.t / second, np.mean(syn_mon[synapses[:, :, 1]].Y_S / umole, axis=0),
           '-', color='darkgreen')
ax[2].set(ylabel=r'$\langle Y_S \rangle$ ($\mu$M)')
ax[2].legend(['open-loop gliotransmission'], loc='upper left')

ax[3].plot(syn_mon.t / second, np.mean(syn_mon[synapses[:, :, 0]].Y_S / umole, axis=0),
           '-', color='red')
ax[3].set(ylabel=r'$\langle Y_S \rangle$ ($\mu$M)', xlabel='time (s)')
ax[3].legend(['closed-loop gliotransmission'], loc='upper left')

# Save Figure for paper # DELETE
#plt.savefig('../text/figures/results/example_4_Figure.png')  # DELETE
plt.show()
