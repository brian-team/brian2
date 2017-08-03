#import matplotlib  # DELETE
#matplotlib.use('agg')  # DELETE
from brian2 import *

GSL=True
if GSL:
    neuron_method = 'GSL_stateupdater'
    synapse_method = 'GSL_stateupdater'
else:
    neuron_method = 'rk4'
    synapse_method = None

# set_device('cpp_standalone', directory='astro_ring')  # uncomment for fast simulation

################################################################################
# Model parameters
################################################################################
### General parameters
duration = 4000*second
dt = 50*ms               # integrator/sampling step

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
F = 0.09*umole/second       # GJC IP_3 permeability (nonlinear)
I_Theta = 0.3*umole      # Threshold IP_3 gradient for diffusion
omega_I = 0.05*umole     # Scaling factor of diffusion
# I_bias = 0.0 * umole     # External IP_3 drive

################################################################################
# Model definition
################################################################################
defaultclock.dt = dt  # Set the integration time

### Astrocytes
astro_eqs = '''
dI/dt = J_delta - J_3K - J_5P + I_exogenous + I_coupling : mole
J_delta = O_delta/(1 + I/kappa_delta) * C**2/(C**2 + K_delta**2) : mole/second
J_3K = O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K) : mole/second
J_5P = Omega_5P*I : mole/second
# Exogenous stimulation (rectangular wave with period of 50s and duty factor 0.4)
stimulus = int((t % (50*second))<20*second) : 1
delta_I_bias = I - I_bias*stimulus : mole
I_exogenous = -F/2*(1 + tanh((abs(delta_I_bias) -
                             I_Theta)/omega_I)) *
              sign(delta_I_bias) : mole/second
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

# External IP_3 drive
I_bias : mole (constant)
'''

N_astro = 50 # Total number of astrocytes in the network
astrocytes = NeuronGroup(N_astro, astro_eqs, method=neuron_method)
# Asymmetric stimulation on the 50th cell to get some nice chaotic patterns
astrocytes.I_bias[N_astro//2] = 1.0*umole
astrocytes.h = 0.9
# Diffusion between astrocytes
# INCLUDE BEGIN
astro_to_astro_eqs = '''
delta_I = I_post - I_pre : mole
I_coupling_post = -F/2 * (1 + tanh((abs(delta_I) - I_Theta)/omega_I)) *
                  sign(delta_I) : mole/second (summed)
'''
astro_to_astro = Synapses(astrocytes, astrocytes,
                          model=astro_to_astro_eqs,
                          method=synapse_method)
# INCLUDE END
# Couple neighbouring astrocytes (we need two connections per astrocyte pair, as
# the above formulation will only update the I_coupling term of one of the
# astrocytes
# INCLUDE BEGIN
astro_to_astro.connect('j == (i + 1) % N_pre or '
                       'j == (i + N_pre - 1) % N_pre')
# INCLUDE END

################################################################################
# Monitors
################################################################################
astro_mon = StateMonitor(astrocytes, variables=['C'], record=True)

################################################################################
# Simulation run
################################################################################
run(duration, report='text')

################################################################################
# Analysis and plotting
################################################################################
plt.style.use('figures.mplstyle')

plt.figure(figsize=(8, 4.5))  # slightly smaller than the default (DELETE)
plt.plot(astro_mon.t/second, (astro_mon.C[0:N_astro//2-1].T/astro_mon.C.max())+np.arange(N_astro//2-1)*1.2, color='black')
plt.plot(astro_mon.t/second, (astro_mon.C[N_astro//2:].T/astro_mon.C.max())+np.arange(N_astro//2,N_astro)*1.2, color='black')
plt.plot(astro_mon.t/second, (astro_mon.C[N_astro//2-1].T/astro_mon.C.max())+np.arange(N_astro//2-1,N_astro//2)*1.2, color='red')
plt.ylabel('$C/C_{max}$')
plt.xlabel('time (s)')
plt.tight_layout()

# Save Figure for paper # DELETE
#plt.savefig('../text/figures/results/example_5_astro_ring_Figure.png')  # DELETE
plt.savefig('example_5_%s.pdf'%('GSL'*GSL+'conventional'*(not GSL)))
plt.show()
