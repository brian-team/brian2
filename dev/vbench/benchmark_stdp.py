from vbench.benchmark import Benchmark

common_setup = """
from brian2 import *
"""

numpy_setup = """
brian_prefs.codegen.target = 'numpy'
"""

weave_setup = """
brian_prefs.codegen.target = 'weave'
"""

setup= """
N = 1000
taum = 10 * ms
taupre = 20 * ms
taupost = taupre
Ee = 0 * mV
vt = -54 * mV
vr = -60 * mV
El = -74 * mV
taue = 5 * ms
F = 15 * Hz
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

eqs_neurons = '''
dv/dt=(ge*(Ee-vr)+El-v)/taum : volt   # the synaptic current is linearized
dge/dt=-ge/taue : 1
'''

input = PoissonGroup(N, rates=F)
neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v=vr')
S = Synapses(input, neurons,
             '''w:1
                dApre/dt=-Apre/taupre : 1 (event-driven)
                dApost/dt=-Apost/taupost : 1 (event-driven)''',
             pre='''ge+=w
                    Apre+=dApre
                    w=clip(w+Apost,0,gmax)''',
             post='''Apost+=dApost
                     w=clip(w+Apre,0,gmax)''',
             connect=True,
             )
S.w='rand()*gmax'
"""

statement = '''
run(100 * second)
'''

bench_stdp_numpy = Benchmark(statement, common_setup + numpy_setup + setup,
                             name='Exponential STDP (numpy)')

bench_stdp_weave = Benchmark(statement, common_setup + weave_setup + setup,
                             name='Exponential STDP (weave)')
