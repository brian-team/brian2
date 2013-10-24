standalone_mode = True
plot_results = True

from pylab import *
from numpy import *
from brian2 import *
import time
import shutil, os

#BrianLogger.log_level_debug()

start = time.time()

if standalone_mode:
    from brian2.devices.cpp_standalone import *
    set_device('cpp_standalone')
else:
    brian_prefs['codegen.target'] = 'weave'
    #brian_prefs['codegen.target'] = 'numpy'

##### Define the model
tau = 1*ms
eqs = '''
dV/dt = (-40*mV-V)/tau : volt (unless refractory)
u : 1
'''
threshold = 'V>-50*mV'
reset = 'V=-60*mV'
refractory = 5*ms
N = 1000
G = NeuronGroup(N, eqs,
                reset=reset,
                threshold=threshold,
                refractory=refractory,
                name='gp')
G.V['i>500'] = '-i*mV'
if standalone_mode:
    arr2d = cpp_standalone_device.dynamic_array(G, 'test', (10, 10), 1., dtype=float)
G.u[[1, 2]] = [3.14, 2.78]
G.u[array([3, 4])] = array([1.41, 6.66])
M = SpikeMonitor(G)
R = PopulationRateMonitor(G, name='ratemon')
S = Synapses(G, G, 'w : volt', pre='V += w')
S.connect('abs(i-j)<5 and i!=j')
S.w = 0.5*mV
S.delay = '0*ms'
net = Network(G,
              M,
              S,
              R,
              )


##### Generate/run code
if not standalone_mode:
    net.run(0*ms)
    start_sim = time.time()

#net.run(100*ms)
run(100*ms)

#net.remove(M, S)

#net.run(10*ms)

insert_device_code('main.cpp', '''
cout << "Testing direct insertion of code." << endl;
''')

if standalone_mode:
    if os.path.exists('output'):
        shutil.rmtree('output')
    build(project_dir='output', compile_project=True, run_project=True)
    print 'Build time:', time.time()-start
    u = loadtxt('output/results/_array_gp_u.txt', delimiter=',', dtype=float)
    print 'G.u[:5] =', u[:5]
    if plot_results:
        subplot(211)
        S = loadtxt('output/results/spikemonitor_codeobject.txt', delimiter=',',
                    dtype=[('i', int), ('t', float)])
        i = S['i']
        t = S['t']*second
        plot(t, i, '.k')
        subplot(212)
        S = loadtxt('output/results/ratemon_codeobject.txt', delimiter=',',
                    dtype=[('t', float), ('rate', float)])
        t = S['t']*second
        rate = S['rate']*Hz
        plot(t, rate)
else:
    print 'Build time:', start_sim-start
    print 'Simulation time:', time.time()-start_sim
    print 'Num spikes:', sum(M.count)
    print 'Num synapses:', len(S)
    print 'G.u[:5] =', G.u[:5]
    if plot_results:
        subplot(211)
        i, t = M.it
        plot(t, i, '.k')
        subplot(212)
        plot(R.t, R.rate)

if plot_results:
    show()
