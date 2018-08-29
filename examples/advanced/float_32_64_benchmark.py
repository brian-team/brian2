'''
Benchmark showing the performance of float32 versus float64.
'''

from brian2 import *
from brian2.devices.device import reset_device, reinit_devices

# CUBA benchmark
def run_benchmark(name):
    if name=='CUBA':

        taum = 20*ms
        taue = 5*ms
        taui = 10*ms
        Vt = -50*mV
        Vr = -60*mV
        El = -49*mV

        eqs = '''
        dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
        dge/dt = -ge/taue : volt
        dgi/dt = -gi/taui : volt
        '''

        P = NeuronGroup(4000, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms,
                        method='exact')
        P.v = 'Vr + rand() * (Vt - Vr)'
        P.ge = 0*mV
        P.gi = 0*mV

        we = (60*0.27/10)*mV # excitatory synaptic weight (voltage)
        wi = (-20*4.5/10)*mV # inhibitory synaptic weight
        Ce = Synapses(P, P, on_pre='ge += we')
        Ci = Synapses(P, P, on_pre='gi += wi')
        Ce.connect('i<3200', p=0.02)
        Ci.connect('i>=3200', p=0.02)

    elif name=='COBA':

        # Parameters
        area = 20000 * umetre ** 2
        Cm = (1 * ufarad * cm ** -2) * area
        gl = (5e-5 * siemens * cm ** -2) * area

        El = -60 * mV
        EK = -90 * mV
        ENa = 50 * mV
        g_na = (100 * msiemens * cm ** -2) * area
        g_kd = (30 * msiemens * cm ** -2) * area
        VT = -63 * mV
        # Time constants
        taue = 5 * ms
        taui = 10 * ms
        # Reversal potentials
        Ee = 0 * mV
        Ei = -80 * mV
        we = 6 * nS  # excitatory synaptic weight
        wi = 67 * nS  # inhibitory synaptic weight

        # The model
        eqs = Equations('''
        dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
                 g_na*(m*m*m)*h*(v-ENa)-
                 g_kd*(n*n*n*n)*(v-EK))/Cm : volt
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        dge/dt = -ge*(1./taue) : siemens
        dgi/dt = -gi*(1./taui) : siemens
        alpha_m = 0.32*(mV**-1)*(13*mV-v+VT)/
                 (exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
        beta_m = 0.28*(mV**-1)*(v-VT-40*mV)/
                (exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
        alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
        alpha_n = 0.032*(mV**-1)*(15*mV-v+VT)/
                 (exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
        beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
        ''')

        P = NeuronGroup(4000, model=eqs, threshold='v>-20*mV', refractory=3 * ms,
                        method='exponential_euler')
        Pe = P[:3200]
        Pi = P[3200:]
        Ce = Synapses(Pe, P, on_pre='ge+=we')
        Ci = Synapses(Pi, P, on_pre='gi+=wi')
        Ce.connect(p=0.02)
        Ci.connect(p=0.02)

        # Initialization
        P.v = 'El + (randn() * 5 - 5)*mV'
        P.ge = '(randn() * 1.5 + 4) * 10.*nS'
        P.gi = '(randn() * 12 + 20) * 10.*nS'

    run(1 * second, profile=True)

    return sum(t for name, t in magic_network.profiling_info)

def generate_results(num_repeats):
    results = {}

    for name in ['CUBA', 'COBA']:
        for target in ['numpy', 'cython', 'weave']:
            for dtype in [float32, float64]:
                prefs.codegen.target = target
                prefs.core.default_float_dtype = dtype
                times = [run_benchmark(name) for repeat in range(num_repeats)]
                results[name, target, dtype.__name__] = amin(times)

    for name in ['CUBA', 'COBA']:
        for dtype in [float32, float64]:
            times = []
            for _ in range(num_repeats):
                reset_device()
                reinit_devices()
                set_device('cpp_standalone', directory=None, with_output=False)
                prefs.core.default_float_dtype = dtype
                times.append(run_benchmark(name))
            results[name, 'cpp_standalone', dtype.__name__] = amin(times)

    return results

results = generate_results(3)

bar_width = 0.9
names = ['CUBA', 'COBA']
targets = ['numpy', 'cython', 'weave', 'cpp_standalone']
precisions = ['float32', 'float64']

figure(figsize=(8, 8))
for j, name in enumerate(names):
    subplot(2, 2, 1+2*j)
    title(name)
    index = arange(len(targets))
    for i, precision in enumerate(precisions):
        bar(index+i*bar_width/len(precisions),
            [results[name, target, precision] for target in targets],
            bar_width/len(precisions), label=precision, align='edge')
    ylabel('Time (s)')
    if j:
        xticks(index+0.5*bar_width, targets, rotation=45)
    else:
        xticks(index+0.5*bar_width, ('',)*len(targets))
        legend(loc='best')

    subplot(2, 2, 2+2*j)
    index = arange(len(precisions))
    for i, target in enumerate(targets):
        bar(index+i*bar_width/len(targets),
            [results[name, target, precision] for precision in precisions],
            bar_width/len(targets), label=target, align='edge')
    ylabel('Time (s)')
    if j:
        xticks(index+0.5*bar_width, precisions, rotation=45)
    else:
        xticks(index+0.5*bar_width, ('',)*len(precisions))
        legend(loc='best')

tight_layout()
show()
