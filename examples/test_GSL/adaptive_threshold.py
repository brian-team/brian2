'''
A model with adaptive threshold (increases with each spike)
'''
from brian2 import *
from brian2.devices.cpp_standalone import GSLCPPStandaloneCodeObject

set_device('cpp_standalone', directory='adaptive_threshold_cpp')

brian = False

eqs = '''
dv/dt = -v/(10*ms) : volt
dvt/dt = (10*mV-vt)/(15*ms) : volt
'''

reset = '''
v = 0*mV
vt += 3*mV
'''

if brian:
    ### Then without (so we can check if same/similar results
    seed(0)

    IF_linear = NeuronGroup(1, model=eqs, reset=reset, threshold='v>vt',
                     method='linear')
    IF_linear.vt = 10*mV

    PG = PoissonGroup(1, 500 * Hz)

    C = Synapses(PG, IF_linear, on_pre='v += 3*mV')
    C.connect()

    Mv_linear = StateMonitor(IF_linear, 'v', record=True)
    Mvt_linear = StateMonitor(IF_linear, 'vt', record=True)
    # Record the value of v when the threshold is crossed
    M_crossings_linear = SpikeMonitor(IF_linear, variables='v')

    linearnet = Network(IF_linear, PG, C, Mv_linear, Mvt_linear, M_crossings_linear)
    linearnet.run(2*second, report='text')
    #print IF_linear.state_updater.codeobj.code


### Run with GSL
seed(0)

IF_GSL = NeuronGroup(1, model=eqs, reset=reset, threshold='v>vt',
                 method='GSL_stateupdater')

IF_GSL.state_updater.codeobj_class = GSLCPPStandaloneCodeObject
IF_GSL.vt = 10*mV

PG = PoissonGroup(1, 500 * Hz)

C = Synapses(PG, IF_GSL, on_pre='v += 3*mV')
C.connect()

Mv_GSL = StateMonitor(IF_GSL, 'v', record=True)
Mvt_GSL = StateMonitor(IF_GSL, 'vt', record=True)
# Record the value of v when the threshold is crossed
M_crossings_GSL = SpikeMonitor(IF_GSL, variables='v')

GSLnet = Network(IF_GSL, PG, C, Mv_GSL, Mvt_GSL, M_crossings_GSL)
GSLnet.run(2*second, report='text')

print IF_GSL.state_updater.codeobj.code

subplot(1, 3, 1)
plot(Mv_GSL.t / ms, Mv_GSL[0].v / mV, label='GSL (rk4)')
plot(Mvt_GSL.t / ms, Mvt_GSL[0].vt / mV, label='GSL (rk4)')
try:
    plot(Mv_linear.t / ms, Mv_linear[0].v / mV, '--', label='Brian (linear)')
    plot(Mvt_linear.t / ms, Mvt_linear[0].vt / mV, '--', label='Brian (linear)')
except NameError:
    pass
ylabel('v (mV)')
xlabel('t (ms)')
title('V and Vt traces\nBrian and GSL')
# zoom in on the first 100ms
xlim(0, 100)
subplot(1, 3, 2)
hist(M_crossings_GSL.v / mV, bins=np.arange(10, 20, 0.5))
xlabel('v at threshold\ncrossing (mV)')
title('Histogram GSL')

subplot(1, 3, 3)
try:
    hist(M_crossings_linear.v / mV, bins=np.arange(10, 20, 0.5))
except NameError:
    pass
xlabel('v at thresholdcrossing (mV)')
title('Histogram Brian')
show()
