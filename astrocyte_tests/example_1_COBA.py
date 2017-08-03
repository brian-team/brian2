#import matplotlib  # DELETE
#matplotlib.use('agg')  # DELETE

from brian2 import *

GSL=True
if GSL:
    neuron_method = 'GSL_stateupdater'
    synapse_method = 'GSL_stateupdater'
else:
    neuron_method = 'euler'
    synapse_method = None

seed(11922)  # to get identical figures for repeated runs

################################################################################
# Model parameters
################################################################################
### General parameters
duration = 1.0*second  # Total simulation time
dt = 0.1*ms            # integrator/sampling step
N_e = 3200           # number of excitatory neurons
N_i = 800            # number of inhibitory neurons

### Neuron parameters
E_l = -60*mV         # Leak reversal potential
g_l = 9.99*nS          # Leak conductance
E_e = 0*mV           # Excitatory reversal potential
E_i = -80*mV         # Inhibitory reversal potential
C_m = 198*pF         # Membrane capacitance
tau_e = 5*ms         # Excitatory time constant
tau_i = 10*ms        # Inhibitory time constant
tau_r = 5*ms         # Refractory time
I_ex = 150*pA        # constant current input
V_th = -50*mV        # Threshold
V_r = E_l            # Reset potential

### Synapse parameters
w_e = 0.05*nS         # excitatory synaptic weight
w_i = 1.0*nS          # inhibitory synaptic weight
U_0 = 0.6             # synaptic release probability
Omega_d = 2.0/second  # Depression rate
Omega_f = 3.33/second # Facilitation rate

################################################################################
# Model definition
################################################################################
# Set the integration time (in this case not strictly necessary, since we are
# using the default value)
defaultclock.dt = dt

### Neurons
# INCLUDE BEGIN
neuron_eqs = '''
dv/dt = (g_l*(E_l-v) +
         g_e*(E_e-v) + g_i*(E_i-v) +
         I_ex)/C_m : volt (unless refractory)
dg_e/dt = -g_e/tau_e : siemens  # post-synaptic exc. conductance
dg_i/dt = -g_i/tau_i : siemens  # post-synaptic inh. conductance
'''
# INCLUDE END
# INCLUDE BEGIN
neurons = NeuronGroup(N_e + N_i, model=neuron_eqs,
                      threshold='v>V_th', reset='v=V_r',
                      refractory='tau_r', method=neuron_method)
# INCLUDE END
# Random initial membrane potential values and conductances
# INCLUDE BEGINapwidths_p4.py
neurons.v = 'E_l + rand()*(V_th-E_l)'
neurons.g_e = 'rand()*w_e'
neurons.g_i = 'rand()*w_i'
exc_neurons = neurons[:N_e]
inh_neurons = neurons[N_e:]
# INCLUDE END
### Synapses
# INCLUDE BEGIN
synapses_eqs = '''
# Usage of releasable neurotransmitter per single action potential:
du_S/dt = -Omega_f * u_S : 1 (event-driven)
# Fraction of synaptic neurotransmitter resources available:
dx_S/dt = Omega_d *(1 - x_S) : 1 (event-driven)
'''
# INCLUDE END
# INCLUDE BEGIN
synapses_action = '''
u_S += U_0 * (1 - u_S)
r_S = u_S * x_S
x_S -= r_S
'''
# INCLUDE END
# INCLUDE BEGIN
exc_syn = Synapses(exc_neurons, neurons, model=synapses_eqs,
                   on_pre=synapses_action+'g_e_post += w_e*r_S',
                   method=synapse_method)
inh_syn = Synapses(inh_neurons, neurons, model=synapses_eqs,
                   on_pre=synapses_action+'g_i_post += w_i*r_S',
                   method=synapse_method)

# INCLUDE END
# INCLUDE BEGIN
exc_syn.connect(True, p=0.05)
inh_syn.connect(True, p=0.2)
# INCLUDE END
### Start from "resting" condition: all synapses have fully-replenished neurotransmitter resources
exc_syn.x_S = 1
inh_syn.x_S = 1

################################################################################
# Monitors
################################################################################
# Note that we could use a single monitor for all neurons instead, but this
# way plotting is a bit easier in the end
exc_mon = SpikeMonitor(exc_neurons)
inh_mon = SpikeMonitor(inh_neurons)

### We record some additional data from a single excitatory neuron
ni = 50
state_mon = StateMonitor(exc_neurons, ['v', 'g_e', 'g_i'], record=ni)  # Record conductances and membrane potential of a given neuron
# We make sure to monitor synaptic variables after synapse are updated in order to use simple recurrence relations to
# reconstruct them
synapse_mon = StateMonitor(exc_syn, ['u_S', 'x_S'],
                           record=exc_syn[ni, :], when='after_synapses')  # Record synapses originating from neuron ni

################################################################################
# Simulation run
################################################################################
run(duration, report='text')

################################################################################
# Analysis and plotting
################################################################################
plt.style.use('figures.mplstyle')

### Spiking activity (w/ rate)
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})
ax[0].plot(exc_mon.t[exc_mon.i <= N_e//4]/ms, exc_mon.i[exc_mon.i <= N_e//4], '.', color='darkred')
ax[0].plot(inh_mon.t[inh_mon.i <= N_i//4]/ms, inh_mon.i[inh_mon.i <= N_i//4]+N_e//4, '.', color='darkblue')
ax[0].set(ylabel='neuron index')

# Generate frequencies
bin_size = 1*ms
spk_count, bin_edges = np.histogram(np.r_[exc_mon.t/ms, inh_mon.t/ms],
                                    int(duration/bin_size))
rate = 1.0*spk_count/(N_e + N_i)/bin_size/Hz
ax[1].plot(bin_edges[:-1], rate,'-', color='k')
ax[1].set(ylim=(0, np.max(rate)), xlabel='time (ms)', ylabel='rate (Hz)')

# Save Figure for paper # DELETE
#savefig('../text/figures/results/example_1_COBA_Figure_1.png')  # DELETE
show()

### Dynamics of a single neuron
fig, ax = plt.subplots(4, sharex=True)
### Postsynaptic conductances
ax[0].plot(state_mon.t/ms, state_mon[ni].g_e/nS, color='darkred')
ax[0].plot(state_mon.t/ms, -state_mon[ni].g_i/nS, color='darkblue')
ax[0].plot([state_mon.t[0]/ms, state_mon.t[-1]/ms], [0, 0], color='grey', linestyle=':')
ax[0].set_ylabel('postsynaptic\nconductance\n(${0}$)'.format(sympy.latex(nS)),
                 multialignment='center')

### Membrane potential
ax[1].plot(state_mon.t/ms, state_mon[ni].v/mV, lw=2, color='black')
ax[1].axhline(V_r/mV, color='purple', linestyle=':')  # Reset potential
ax[1].axhline(V_th/mV, color='orange', linestyle=':')  # Threshold
# Artificially insert spikes
ax[1].vlines(exc_mon.t[exc_mon.i == ni]/ms, -50, 0, color='black')
ax[1].set_ylabel('membrane\npotential\n(${0}$)'.format(sympy.latex(mV)),
                 multialignment='center')

### Synaptic variables
# Retrieves indexes of spikes in the synaptic monitor using the fact that we are sampling spikes and synaptic variables
# by the same dt
spk_index = np.in1d(synapse_mon.t, exc_mon.t[exc_mon.i == ni])
ax[2].plot(synapse_mon.t[spk_index]/ms, synapse_mon.x_S[0][spk_index], '.',
           ms=8, color='magenta')
ax[2].plot(synapse_mon.t[spk_index]/ms, synapse_mon.u_S[0][spk_index], '.',
           ms=8, color='green')
# Super-impose reconstructed solutions
time = synapse_mon.t/second # Dimensionless copy of time vector
tspk = synapse_mon.t/second # Dimensionless vector of spike times
for ts in exc_mon.t[exc_mon.i == ni]/second:
    tspk[(synapse_mon.t/second)>=ts] = ts
ax[2].plot(synapse_mon.t/ms, 1 + (synapse_mon.x_S[0]-1)*exp(-(time-tspk)*Omega_d*second),
           '-', color='magenta')
ax[2].plot(synapse_mon.t/ms, synapse_mon.u_S[0]*exp(-(time-tspk)*Omega_f*second),
           '-', color='green')
ax[2].set_ylabel('synaptic\nvariables\n$u_S,\,x_S$', multialignment='center')

## This will not show if no spike is fired by the selected neuron
nspikes = np.sum(spk_index)
ax[3].vlines(synapse_mon.t[spk_index]/ms,np.zeros(nspikes),
             synapse_mon.x_S[0][spk_index]*synapse_mon.u_S[0][spk_index],
             color='black')
ax[3].set(ylabel='$r_S$', xlabel='time (ms)')
plt.tight_layout()

# Save Figure for paper # DELETE
plt.savefig('example_1_%s.pdf'%('GSL'*GSL+'conventional'*(not GSL)))
show()
