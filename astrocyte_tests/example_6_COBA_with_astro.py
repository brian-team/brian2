#import matplotlib  # DELETE
#matplotlib.use('agg')  # DELETE
from brian2 import *

GSL=True
if GSL:
    neuron_method = 'GSL_stateupdater'
    synapse_method = 'GSL_stateupdater'
    astro_method = 'GSL_stateupdater'
else:
    neuron_method = 'euler'
    synapse_method = 'linear'
    astro_method = 'rk4'

set_device('cpp_standalone', directory='COBA_with_astro')  # uncomment for fast simulation

seed(2871)  # to get identical figures for repeated runs

################################################################################
# Model parameters
################################################################################
### General parameters
N_e = 3200  # number of excitatory neurons
N_i = 800   # number of inhibitory neurons
N_a = 3200  # number of astrocytes
N_e = 320  # number of excitatory neurons
N_i = 80   # number of inhibitory neurons
N_a = 320  # number of astrocytes

## Some metrics parameters needed to establish proper connections
size = 3.75*mmeter
distance = 50*umeter

### Neuron parameters
E_l = -60*mV    # Leak reversal potential
g_l = 9.99*nS     # Leak conductance
E_e = 0*mV      # Excitatory reversal potential
E_i = -80*mV    # Inhibitory reversal potential
C_m = 198*pF    # Membrane capacitance
tau_e = 5*ms    # Excitatory time constant
tau_i = 10*ms   # Inhibitory time constant
tau_r = 5*ms    # Refractory time
I_ex = 100*pA   # constant current input
V_th = -50*mV   # Threshold
V_r = E_l       # Reset potential

### Synapse parameters
rho_c = 0.0005              # synaptic vesicle-to-extracellular space volume ratio
Y_T = 500.*mmole            # Total neurotransmitter synaptic resource (in terms of vesicular concentration)
Omega_c = 40/second         # Neurotransmitter clearance rate
U_0__star = 0.6             # Basal synaptic release probability
Omega_f = 2.0/second        # Facilitation rate
Omega_d = 3.33/second       # Depression rate
w_e = 0.05*nS               # excitatory synaptic weight
w_i = 1.0*nS                # inhibitory synaptic weight
# --- Presynaptic receptors
O_G = 1.5/umole/second      # Agonist binding rate (activating)
Omega_G = 0.5/(60*second)   # Agonist release rate (inactivating)

### Astrocyte parameters
# ---  Calcium fluxes
O_P = 0.9*umole/second  # Maximal Ca^2+ uptake rate
K_P = 0.05*umole        # Ca2+ affinity of SERCAs
C_T = 2*umole           # Total ER Ca^2+ content
rho_A = 0.18            # ER-to-cytoplasm volume ratio
Omega_C = 6/second      # Maximal Ca^2+ release rate by IP_3Rs
Omega_L = 0.1/second    # Maximal Ca^2+ leak rate,
# --- IP_3R kinectics
d_1 = 0.13*umole        # IP_3 binding affinity
d_2 = 1.05*umole        # Inactivating Ca^2+ binding affinity
O_2 = 0.2/umole/second  # Inactivating Ca^2+ binding rate
d_3 = 0.9434*umole      # IP_3 binding affinity (with Ca^2+ inactivation)
d_5 = 0.08*umole        # Activating Ca^2+ binding affinity
# --- IP_3 production
# --- Agonist-dependent IP_3 production
O_beta = 0.5*umole/second  # Maximal rate of IP_3 production by PLCbeta
O_N = 0.3/umole/second     # Agonist binding rate
Omega_N = 0.5/second       # Inactivation rate of GPCR signalling
K_KC = 0.5*umole           # Ca^2+ affinity of PKC
zeta = 10                  # Maximal reduction of receptor affinity by PKC
# --- Endogenous IP3 production
O_delta = 1.2*umole/second # Maximal rate of IP_3 production by PLCdelta
kappa_delta = 1.5*umole    # Antagonizing IP_3 affinity
K_delta = 0.1*umole        # Ca^2+ affinity of PLCdelta
# --- IP_3 diffusion
F = 2*umole/second      # GJC IP_3 permeability (nonlinear)
I_Theta = 0.3*umole     # Threshold IP_3 gradient for diffusion
omega_I = 0.05*umole    # Scaling factor of diffusion
# --- IP_3 degradation
Omega_5P = 0.05/second  # Maximal rate of IP_3 degradation by IP-5P
K_D = 0.7*umole         # Ca^2+ affinity of IP3-3K
K_3K = 1.0*umole        # IP_3 affinity of IP_3-3K
O_3K = 4.5*umole/second # Maximal rate of IP_3 degradation by IP_3-3K
# --- IP_3 diffusion
F = 0.09*umole/second   # GJC IP_3 permeability (nonlinear)
I_Theta = 0.3*umole     # Threshold IP_3 gradient for diffusion
omega_I = 0.05*umole    # Scaling factor of diffusion
# --- Gliotransmitter release and time course
C_Theta = 0.5*umole     # Ca^2+ threshold for exocytosis
Omega_A = 0.6/second    # Gliotransmitter recycling rate
U_A = 0.6               # Gliotransmitter release probability
G_T = 200*mmole         # Total vesicular gliotransmitter
rho_e = 6.5e-4          # Ratio of astrocytic vesicle volume/ESS volume
Omega_e = 60/second     # Gliotransmitter clearance rate
alpha = 0.0             # Gliotranmission type (assumed maximally inhibitory)

################################################################################
# Define HF stimulus
################################################################################
stimulus = TimedArray([1., 1., 1.2, 1.2, 1.0, 1.0, 1.0, 1.0], dt=1*second)

################################################################################
# Simulation time (based on the stimulus)
################################################################################
duration = 8*second

################################################################################
# Model definition
################################################################################
### Neurons
# INCLUDE BEGIN
neuron_eqs = '''
# ELLIPSIS BEGIN
dv/dt = (g_l*(E_l-v) + g_e*(E_e-v) + g_i*(E_i-v) + I_ex*stimulus(t))/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens  # post-synaptic excitatory conductance
dg_i/dt = -g_i/tau_i : siemens  # post-synaptic inhibitory conductance
# ELLIPSIS END
# Neuron position in space
x : meter (constant)
y : meter (constant)
'''
neurons = NeuronGroup(N_e + N_i, model=neuron_eqs,
                      threshold='v>V_th', reset='v=V_r',
                      refractory='tau_r', method=neuron_method)
exc_neurons = neurons[:N_e]
inh_neurons = neurons[N_e:]
# Arrange excitatory neurons in a grid
N_rows = int(sqrt(N_e))
N_cols = N_e/N_rows
grid_dist = (size / N_cols)
exc_neurons.x = '(i / N_rows)*grid_dist - N_rows/2.0*grid_dist'
exc_neurons.y = '(i % N_rows)*grid_dist - N_cols/2.0*grid_dist'
# INCLUDE END
# Random initial membrane potential values and conductances
neurons.v = 'E_l + rand()*(V_th-E_l)'
neurons.g_e = 'rand()*w_e'
neurons.g_i = 'rand()*w_i'

### Synapses
# INCLUDE BEGIN
synapses_eqs = '''
# ELLIPSIS BEGIN
# Neurotransmitter
dY_S/dt = -Omega_c * Y_S : mole (clock-driven)
# Fraction of activated presynaptic receptors
dGamma_S/dt = O_G * G_A * (1 - Gamma_S) - Omega_G * Gamma_S : 1 (clock-driven)
# Usage of releasable neurotransmitter per single action potential:
du_S/dt = -Omega_f * u_S : 1 (event-driven)
# Fraction of synaptic neurotransmitter resources available for release:
dx_S/dt = Omega_d *(1 - x_S) : 1 (event-driven)
U_0 : 1
r_S : 1  # released synaptic neurotransmitter resources
G_A : mole  # gliotransmitter concentration in the extracellular space
# ELLIPSIS END
# which astrocyte covers this synapse ?
astrocyte_index : integer (constant)
'''
# ELLIPSIS BEGIN
synapses_action = '''
U_0 = (1 - Gamma_S) * U_0__star + alpha * Gamma_S
u_S += U_0 * (1 - u_S)
r_S = u_S * x_S
x_S -= r_S
Y_S += rho_c * Y_T * r_S
'''
# ELLIPSIS END
exc_syn = Synapses(exc_neurons, neurons, model=synapses_eqs,
                   on_pre=synapses_action+'g_e_post += w_e*r_S',
                   method=synapse_method, name='exc_syn')
# INCLUDE END
exc_syn.connect(True, p=0.05)
exc_syn.x_S = 1.0
inh_syn = Synapses(inh_neurons, neurons, model=synapses_eqs,
                   on_pre=synapses_action+'g_i_post += w_i*r_S',
                   method=synapse_method, name='inh_syn')
inh_syn.connect(True, p=0.2)
inh_syn.x_S = 1.0
# Connect excitatory synapses to an astrocyte depending on the position of the
# post-synaptic neuron
# INCLUDE BEGIN
N_rows = int(sqrt(N_a))
N_cols = N_a/N_rows
grid_dist = size / N_rows
exc_syn.astrocyte_index = ('int(x_post/grid_dist) + '
                           'N_cols*int(y_post/grid_dist)')
# INCLUDE END
### Astrocytes
# The astrocyte emits gliotransmitter when its Ca^2+ concentration crosses
# a threshold
# INCLUDE BEGIN
astro_eqs = '''
# ELLIPSIS BEGIN
# Fraction of activated astrocyte receptors:
dGamma_A/dt = O_N * Y_S * (1 - clip(Gamma_A,0,1)) - Omega_N*(1 + zeta * C/(C + K_KC)) * clip(Gamma_A,0,1) : 1
# Intracellular IP_3
dI/dt = J_beta + J_delta - J_3K - J_5P + I_coupling : mole
J_beta = O_beta * Gamma_A : mole/second
J_delta = O_delta/(1 + I/kappa_delta) * C**2/(C**2 + K_delta**2) : mole/second
J_3K = O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K) : mole/second
J_5P = Omega_5P*I : mole/second
# Diffusion between astrocytes:
I_coupling : mole/second

# Ca^2+-induced Ca^2+ release:
dC/dt = J_r + J_l - J_p : mole
dh/dt = (h_inf - h)/tau_h : 1  # IP3R de-inactivation probability
J_r = (Omega_C * m_inf**3 * h**3) * (C_T - (1 + rho_A)*C) : mole/second
J_l = Omega_L * (C_T - (1 + rho_A)*C) : mole/second
J_p = O_P * C**2/(C**2 + K_P**2) : mole/second
m_inf = I/(I + d_1) * C/(C + d_5) : 1
h_inf = Q_2/(Q_2 + C) : 1
tau_h = 1/(O_2 * (Q_2 + C)) : second
Q_2 = d_2 * (I + d_1)/(I + d_3) : mole

dx_A/dt = Omega_A * (1 - x_A) : 1  # Fraction of gliotransmitter resources available for release
dG_A/dt = -Omega_e*G_A : mole  # gliotransmitter concentration in the extracellular space
# Neurotransmitter concentration in the extracellular space
Y_S : mole
# ELLIPSIS END
# The astrocyte position in space
x : meter (constant)
y : meter (constant)
'''
# ELLIPSIS BEGIN
glio_release = '''
G_A += rho_e * G_T * U_A * x_A
x_A -= U_A *  x_A
'''
astrocytes = NeuronGroup(N_a, astro_eqs,
                         # The following formulation makes sure that a "spike" is
                         # only triggered at the first threshold crossing
                         threshold='C>C_Theta',
                         refractory='C>C_Theta',
                         # The gliotransmitter release happens when the threshold
                         # is crossed, in Brian terms it can therefore be
                         # considered a "reset"
                         reset=glio_release,
                         method=astro_method,
                         dt=1e-2*second)
# ELLIPSIS END
# Arrange astrocytes in a grid
astrocytes.x = '(i / N_rows)*grid_dist - N_rows/2.0*grid_dist'
astrocytes.y = '(i % N_rows)*grid_dist - N_cols/2.0*grid_dist'
# INCLUDE END
# Add random initialization
astrocytes.C = 0.01*umole
astrocytes.h = 0.9
astrocytes.I = 0.01*umole
astrocytes.x_A = 1.0

# INCLUDE BEGIN
astro_to_syn = Synapses(astrocytes, exc_syn,
                        'G_A_post = G_A_pre : mole (summed)',
                        method=synapse_method)
astro_to_syn.connect('i == astrocyte_index_post')
syn_to_astro = Synapses(exc_syn, astrocytes,
                        'Y_S_post = Y_S_pre/N_incoming : mole (summed)')
syn_to_astro.connect('astrocyte_index_pre == j')
# INCLUDE END
# Diffusion between astrocytes
# INCLUDE BEGIN
astro_to_astro_eqs = '''
delta_I = I_post - I_pre : mole
I_coupling_post = -(1 + tanh((abs(delta_I) - I_Theta)/omega_I))*
                  sign(delta_I)*F/2 : mole/second (summed)
'''
astro_to_astro = Synapses(astrocytes, astrocytes,
                          model=astro_to_astro_eqs,
                          method=synapse_method)
# Connect to all astrocytes less than 75um away
# (~4 connections per astrocyte)
astro_to_astro.connect('i != j and '
                       'sqrt((x_pre-x_post)**2 +'
                       '     (y_pre-y_post)**2) < 75*um')
# INCLUDE END

################################################################################
# Monitors
################################################################################
# Note that we could use a single monitor for all neurons instead, but this
# way plotting is a bit easier in the end
exc_mon = SpikeMonitor(exc_neurons)
inh_mon = SpikeMonitor(inh_neurons)
ast_mon = SpikeMonitor(astrocytes)

################################################################################
# Simulation run
################################################################################
run(duration, report='text')

################################################################################
# Plot of Spiking activity
################################################################################
plt.style.use('figures.mplstyle')
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True,
                        gridspec_kw={'height_ratios': [1, 6, 2],
                                     'top': 0.98})
time_range = np.linspace(0, duration/second, duration/second*100)*second
ax[0].plot(time_range, stimulus(time_range), 'k')
ax[0].axis('off')

## We only plot a fraction of the spikes
fraction = 4
ax[1].plot(exc_mon.t[exc_mon.i <= N_e//fraction]/second,
           exc_mon.i[exc_mon.i <= N_e//fraction], '.', color='darkred')
ax[1].plot(inh_mon.t[inh_mon.i <= N_i//fraction]/second,
           inh_mon.i[inh_mon.i <= N_i//fraction]+N_e//fraction, '.', color='darkblue')
ax[1].plot(ast_mon.t[ast_mon.i <= N_a//fraction]/second,
       ast_mon.i[ast_mon.i <= N_a//fraction]+(N_e+N_i)//fraction, '.', color='green')

ax[1].set(ylim=[0,(N_e+N_i+N_a)//fraction], ylabel='cell index')

# Generate frequencies
bin_size = 1*ms
spk_count, bin_edges = np.histogram(np.r_[exc_mon.t/second, inh_mon.t/second],
                                    int(duration/bin_size))
rate = 1.0*spk_count/(N_e + N_i)/bin_size/Hz
ax[2].plot(bin_edges[:-1], rate,'-', color='k')
ax[2].set(ylim=(0, np.max(rate)), xlabel='time (s)', ylabel='rate (Hz)')

# Save Figure for paper # DELETE
#plt.savefig('../text/figures/results/example_6_COBA_with_astro_Figure.png')  # DELETE
plt.savefig('example_6_%s.pdf'%('GSL'*GSL+'conventional'*(not GSL)))
plt.show()
