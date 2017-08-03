#import matplotlib  # DELETE
#matplotlib.use('agg')  # DELETE

from brian2 import *

GSL=True
if GSL:
    neuron_method = 'GSL_stateupdater'
    synapse_method = 'GSL_stateupdater'
else:
    neuron_method = 'rk4'
    synapse_method = 'euler'

# set_device('cpp_standalone', directory='io_synapse')  # uncomment for fast simulation

################################################################################
# Model parameters
################################################################################
### General parameters
transient = 16.5*second
duration = transient + 600*ms # Total simulation time
dt = 1*ms                     # integrator/sampling step

### Neuron parameters
f_0 = 0.1*Hz             # Spike rate of the "source" neurons

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
O_delta = 0.6*umole/second # Maximal rate of IP_3 production by PLCdelta
kappa_delta = 1.5* umole  # Antagonizing IP_3 affinity
K_delta = 0.1*umole        # Ca^2+ affinity of PLCdelta
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
# (We are only interested in the activity of the synapse, so we replace the
# neurons by trivial "dummy" groups)
spikes = [0, 50, 100, 150, 200,
          300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400]*ms
spikes += transient  # allow for some initial transient
source_neurons = SpikeGeneratorGroup(1, np.zeros(len(spikes)), spikes)
target_neurons = NeuronGroup(1, '')

### Synapses
# Note that the synapse does not actually have any effect on the post-synaptic
# target
# Also note that for easier plotting we do not use the "event-driven" flag here,
# even though the value of u_S and x_S only needs to be updated on the arrival
# of a spike
# INCLUDE BEGIN
synapses_eqs = '''
# Neurotransmitter
dY_S/dt = -Omega_c * Y_S : mole (clock-driven)
# Fraction of activated presynaptic receptors
dGamma_S/dt = O_G * G_A * (1 - Gamma_S) - Omega_G * Gamma_S : 1 (clock-driven)
# Usage of releasable neurotransmitter per single action potential:
du_S/dt = -Omega_f * u_S : 1 (clock-driven)
# Fraction of synaptic neurotransmitter resources available:
dx_S/dt = Omega_d *(1 - x_S) : 1 (clock-driven)
r_S : 1  # released synaptic neurotransmitter resources
# gliotransmitter concentration in the extracellular space:
G_A : mole
'''
# INCLUDE END
# INCLUDE BEGIN
synapses_action = '''
U_0 = (1 - Gamma_S) * U_0__star + alpha * Gamma_S
u_S += U_0 * (1 - u_S)
r_S = u_S * x_S
x_S -= r_S
Y_S += rho_c * Y_T * r_S
'''
synapses = Synapses(source_neurons, target_neurons,
                    model=synapses_eqs, on_pre=synapses_action, method=synapse_method)
# INCLUDE END
# We create three synapses, only the second and third ones are modulated by astrocytes
synapses.connect(True, n=3)

### Astrocytes
# The astrocyte emits gliotransmitter when its Ca^2+ concentration crosses
# a threshold
# INCLUDE BEGIN
astro_eqs = '''
# ELLIPSIS BEGIN
# IP_3 dynamics:
dI/dt = J_delta - J_3K - J_5P + I_exogenous : mole
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
# ELLIPSIS END
# Fraction of gliotransmitter resources available:
dx_A/dt = Omega_A * (1 - x_A) : 1
# gliotransmitter concentration in the extracellular space:
dG_A/dt = -Omega_e*G_A : mole
'''
glio_release = '''
G_A += rho_e * G_T * U_A * x_A
x_A -= U_A *  x_A
'''
# INCLUDE END
# The following formulation makes sure that a "spike" is only triggered at the
# first threshold crossing -- the astrocyte is considered "refractory" (i.e.,
# not allowed to trigger another event) as long as the Ca2+ concentration
# remains above threshold
# The gliotransmitter release happens when the threshold is crossed, in Brian
# terms it can therefore be considered a "reset"
# INCLUDE BEGIN
astrocyte = NeuronGroup(2, astro_eqs,
                        threshold='C>C_Theta',
                        refractory='C>C_Theta',
                        reset=glio_release,
                        method=neuron_method)
# INCLUDE END
# Different length of stimulation
astrocyte.x_A = 1.0
astrocyte.h = 0.9
astrocyte.I = 0.4*umole
astrocyte.I_bias = np.asarray([0.8, 1.25])*umole

# Connection between astrocytes and the second synapse. Note that in this
# special case, where the synapse is only influenced by the gliotransmitter from
# a single astrocyte, the linked variable mechanism could be used instead.
# The mechanism used below is more general and can add the contribution of
# several astrocytes
# INCLUDE BEGIN
astro_to_syn = Synapses(astrocyte, synapses,
                        'G_A_post = G_A_pre : mole (summed)')
# INCLUDE END
# Connect second and third synapse to a different astrocyte
astro_to_syn.connect(j='i+1')

################################################################################
# Monitors
################################################################################
syn_mon = StateMonitor(synapses, variables=['u_S', 'x_S', 'r_S', 'Y_S'],
                   record=np.arange(2))
ast_mon = StateMonitor(astrocyte, variables=['C','G_A'], record=True)

################################################################################
# Simulation run
################################################################################
run(duration, report='text')
print synapses.state_updater.codeobj.code
print astrocyte.state_updater.codeobj.code

################################################################################
# Analysis and plotting
################################################################################
plt.style.use('figures.mplstyle')

fig, ax = plt.subplots(nrows=7, ncols=1,figsize=(8,12),
                       gridspec_kw={'height_ratios': [3, 2, 1, 1, 3, 3, 3],
                                    'top': 0.98, 'bottom': 0.08,
                                    'left': 0.1, 'right': 0.95})

## Ca^2+ traces of the two astrocytes
ax[0].plot((ast_mon.t-transient)/second, ast_mon.C[0]/umole, '-', color='darkgreen')
ax[0].plot((ast_mon.t-transient)/second, ast_mon.C[1]/umole, '-', color='red')
## Add threshold for gliotransmitter release
ax[0].plot(np.asarray([-transient/second,0.0]), np.asarray([C_Theta,C_Theta])/umole, ':', color='gray')
ax[0].set(xticks=[], yticks=[0., 0.4, 0.8, 1.2], ylabel=r'$C$ ($\mu$M)', xlim=[-transient/second, 0.0])

## Gliotransmitter concentration in the extracellular space
ax[1].plot((ast_mon.t-transient)/second, ast_mon.G_A[0]/umole, '-', color='darkgreen')
ax[1].plot((ast_mon.t-transient)/second, ast_mon.G_A[1]/umole, '-', color='red')
ax[1].set(yticks=[0.,50.,100.], ylabel=r'$G_A$ ($\mu$M)', xlabel='time (s)', xlim=[-transient/second, 0.0])

## Turn off one axis to display x-labeling of ax[1] correctly
ax[2].axis('off')

## Synaptic stimulation
ax[3].vlines((spikes-transient)/ms, 0, 1, clip_on=False)
ax[3].set(xlim=(0, (duration-transient)/ms))
ax[3].axis('off')

## Synaptic variables
ax[4].plot((syn_mon.t-transient)/ms, syn_mon.u_S.T)
ax[4].set(xticks=[], ylabel='$u_S$', yticks=[0, .25, .5, .75, 1], xlim=(0, (duration-transient)/ms))

ax[5].plot((syn_mon.t-transient)/ms, syn_mon.x_S.T)
ax[5].set(xticks=[], ylabel='$x_S$', yticks=[0, .25, .5, .75, 1], xlim=(0, (duration-transient)/ms))

ax[6].plot((syn_mon.t-transient)/ms, syn_mon.Y_S.T/umole)
ax[6].set(xlim=(0, (duration-transient)/ms), yticks=[0, 500, 1000, 1500], ylabel=r'$Y_S$ ($\mu$M)', xlabel='time (ms)')
ax[6].legend(['no gliotransmission', 'weak gliotransmission', 'stronger gliotransmission'], loc='upper right')

# Save Figure for paper # DELETE
plt.savefig('example_3_%s.pdf'%('GSL'*GSL+'conventional'*(not GSL)))
plt.show()
