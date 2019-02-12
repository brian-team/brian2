'''
Reproduces Figure 9 from the following paper:
Dendritic Low-Threshold Calcium Currents in Thalamic Relay Cells
Alain Destexhe, Mike Neubig, Daniel Ulrich, John Huguenard
Journal of Neuroscience 15 May 1998, 18 (10) 3574-3588

The original NEURON code is available on ModelDB: https://senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=279 

Reference for the original morphology:
Rat VB neuron (thalamocortical cell), given by J. Huguenard, stained with
biocytin and traced by A. Destexhe, December 1992.  The neuron is described
in: J.R. Huguenard & D.A. Prince, A novel T-type current underlies prolonged
calcium-dependent burst firing in GABAergic neurons of rat thalamic reticular
nucleus.  J. Neurosci. 12: 3804-3817, 1992.

Available at NeuroMorpho.org:
http://neuromorpho.org/neuron_info.jsp?neuron_name=tc200
NeuroMorpho.Org ID :NMO_00881

Notes
-----
* Completely removed the "Fast mechanism for submembranal Ca++ concentration
  (cai)" -- it did not affect the results presented here
* Time constants for the I_T current are slightly different from the equations
  given in the paper -- the paper calculation seems to be based on 36 degree
  Celsius but the temperature that is used is 34 degrees.
* Other small discrepancies with the paper -- values from the NEURON code were
  used whenever different from the values stated in the paper
'''
from __future__ import print_function
from brian2 import *
from brian2.units.constants import (zero_celsius, faraday_constant as F,
                                    gas_constant as R)

defaultclock.dt = 0.01*ms

VT = -52*mV
El = -76.5*mV  # from code, text says: -69.85*mV
gl = 0.0379*msiemens/cm**2

E_Na = 50*mV
E_K = -100*mV

T = 34*kelvin + zero_celsius # 34 degC (current-clamp experiments)
tadj_HH = 3.0**((34-36)/10.0)  # temperature adjustment for Na & K (original recordings at 36 degC)
tadj_m_T = 2.5**((34-24)/10.0)
tadj_h_T = 2.5**((34-24)/10.0)

shift_I_T = -1*mV

gamma = F/(R*T)  # R=gas constant, F=Faraday constant
Z_Ca = 2  # Valence of Calcium ions
Ca_i = 240*nM  # intracellular Calcium concentration
Ca_o = 2*mM  # extracellular Calcium concentration

eqs = Equations('''
Im = gl*(El-v) - I_Na - I_K - I_T: amp/meter**2
I_inj : amp (point current)

# HH-type currents for spike initiation
g_Na : siemens/meter**2
g_K : siemens/meter**2
I_Na = g_Na * m**3 * h * (v-E_Na) : amp/meter**2
I_K = g_K * n**4 * (v-E_K) : amp/meter**2
v2 = v - VT : volt  # shifted membrane potential (Traub convention)
dm/dt = (0.32*(mV**-1)*(13.*mV-v2)/
        (exp((13.*mV-v2)/(4.*mV))-1.)*(1-m)-0.28*(mV**-1)*(v2-40.*mV)/
        (exp((v2-40.*mV)/(5.*mV))-1.)*m) / ms * tadj_HH: 1
dn/dt = (0.032*(mV**-1)*(15.*mV-v2)/
        (exp((15.*mV-v2)/(5.*mV))-1.)*(1.-n)-.5*exp((10.*mV-v2)/(40.*mV))*n) / ms * tadj_HH: 1
dh/dt = (0.128*exp((17.*mV-v2)/(18.*mV))*(1.-h)-4./(1+exp((40.*mV-v2)/(5.*mV)))*h) / ms * tadj_HH: 1

# Low-threshold Calcium current (I_T)  -- nonlinear function of voltage
I_T = P_Ca * m_T**2*h_T * G_Ca : amp/meter**2
P_Ca : meter/second  # maximum Permeability to Calcium
G_Ca = Z_Ca**2*F*v*gamma*(Ca_i - Ca_o*exp(-Z_Ca*gamma*v))/(1 - exp(-Z_Ca*gamma*v)) : coulomb/meter**3
dm_T/dt = -(m_T - m_T_inf)/tau_m_T : 1
dh_T/dt = -(h_T - h_T_inf)/tau_h_T : 1
m_T_inf = 1/(1 + exp(-(v/mV + 56)/6.2)) : 1
h_T_inf = 1/(1 + exp((v/mV + 80)/4)) : 1
tau_m_T = (0.612 + 1.0/(exp(-(v/mV + 131)/16.7) + exp((v/mV + 15.8)/18.2))) * ms / tadj_m_T: second
tau_h_T = (int(v<-81*mV) * exp((v/mV + 466)/66.6) +
           int(v>=-81*mV) * (28 + exp(-(v/mV + 21)/10.5))) * ms / tadj_h_T: second
''')

# Load morphology from SWC file
morpho = Morphology.from_file('tc200.CNG.swc')
neuron = SpatialNeuron(morpho, eqs, Cm=0.88*uF/cm**2, Ri=173*ohm*cm,
                       method='exponential_euler')

neuron.v = -74*mV
# Only the soma has Na/K channels
neuron.main.g_Na = 100*msiemens/cm**2
neuron.main.g_K = 100*msiemens/cm**2

neuron.m_T = 'm_T_inf'
neuron.h_T = 'h_T_inf'

mon = StateMonitor(neuron, ['v'], record=0)  # Record at soma
cutoff = 100*ms  # we'll ignore the first 100ms
store('initial state')


def do_experiment(currents, somatic_density, dendritic_density):
    restore('initial state')
    voltages = []
    neuron.P_Ca = somatic_density
    # Distal dendrites
    neuron.P_Ca['(distance + length/2) > 11*um'] = dendritic_density
    # Switch off monitor during initial transient
    mon.active = False
    run(cutoff)
    mon.active = True
    run(80*ms)
    store('before current')
    for current in currents:
        restore('before current')
        neuron.main.I_inj = current
        print('.', end='')
        run(320*ms)
        voltages.append(mon[morpho].v[:])  # somatic voltage
    return voltages


fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
# Uniform density
voltages = do_experiment([50, 75]*pA, somatic_density=1.7e-5*cm/second,
                         dendritic_density=1.7e-5*cm/second)
axes[0].plot((mon.t - cutoff)/ms, voltages[0]/mV, label='50pA')
axes[0].plot((mon.t - cutoff)/ms, voltages[1]/mV, label='75pA')
axes[0].legend(loc='best', frameon=False)
axes[0].set(xlabel='Time(ms)', ylabel='Voltage (mV)',
            xlim=(0, 400), xticks=[0, 100, 200, 300, 400],
            ylim=(-80, 40), yticks=[-80, -40, 0, 40],
            title='Uniform T-current')
# Increased T-current in dendrites
voltages = do_experiment([50, 75]*pA, somatic_density=1.7e-5*cm/second,
                         dendritic_density=8.5e-5*cm/second)
axes[1].plot((mon.t - cutoff)/ms, voltages[0]/mV, label='50pA')
axes[1].plot((mon.t - cutoff)/ms, voltages[1]/mV, label='75pA')
axes[1].set(title='Increased T-current in dendrites')

for ax in axes:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.show()
