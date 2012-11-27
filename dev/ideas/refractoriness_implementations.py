from pylab import *
from brian2 import *
from scipy import weave
import time

N = 10000
refractory = 5*ms
duration = 100*ms
language = PythonLanguage()
#language = CPPLanguage()

eqs = '''
dv/dt = (ge+gi-(v+49*mV))/(10*ms) : volt
dge/dt = -ge/(5*ms) : volt
dgi/dt = -gi/(10*ms) : volt
tlast : second
'''

reset = '''
v = -60*mV
tlast = t
'''

threshold = 'v>-50*mV'

#P = NeuronGroup(N, eqs, reset='v=-60*mV', threshold='v>-50*mV')

v0 = -60*mV+10*mV*rand(N)

def reinitP(P):
    P.v = v0
    P.tlast = -1*second

def brian_1x():
    eqs2 = eqs+'vref : volt'
    reset2 = reset+'vref = v'
    P = NeuronGroup(N, eqs2, reset=reset2, threshold=threshold,
                    language=language)
    reinitP(P)
    Ptlast = P.tlast_
    refractory_ = float(refractory)
    vref = P.vref_
    v = P.v_
    @network_operation(when=('groups', 100))
    def refrac():
        t = defaultclock.t_
        I, = (t<Ptlast+refractory_).nonzero()
        v[I] = vref[I]
    run(1*ms)
    start = time.time()
    run(duration)
    end = time.time()
    return (end-start, sum(v))

def modify_equations():
    eqs2 = '''
    dv/dt = is_active*(ge+gi-(v+49*mV))/(10*ms) : volt
    dge/dt = -ge/(5*ms) : volt
    dgi/dt = -gi/(10*ms) : volt
    tlast : second
    is_active : 1
    '''
    P = NeuronGroup(N, eqs2, reset=reset, threshold=threshold,
                    language=language)
    actrun = P.runner('is_active = t>=tlast+refractory')
    reinitP(P)
    run(1*ms)
    start = time.time()
    run(duration)
    end = time.time()
    return (end-start, sum(P.v_))

def multiple_updaters():
    eqs2 = '''
    v : volt
    dge/dt = -ge/(5*ms) : volt
    dgi/dt = -gi/(10*ms) : volt
    tlast : second
    '''
    P1 = NeuronGroup(N, eqs, reset=reset, threshold=threshold,
                     language=language)
    P2 = NeuronGroup(N, eqs2, reset=reset, threshold=threshold,
                     language=language)
    su1 = P1.state_updater
    su2 = P2.state_updater
    ns1 = su1.codeobj.namespace
    ns2 = su2.codeobj.namespace
    P1.contained_objects.remove(P1.state_updater)
    assert len(P1.contained_objects)==2
    v = P1.v_
    ge = P1.ge_
    gi = P1.gi_
    tlast = P1.tlast_
    refractory_ = float(refractory)
    @network_operation(when='groups')
    def msu():
        t = defaultclock.t_
        I = t>=tlast+refractory_
        #print sum(I)
        active, = I.nonzero()
        inactive, = (-I).nonzero()
        ns1['_array_v'] = v[active]
        ns1['_array_ge'] = ge[active]
        ns1['_array_gi'] = gi[active]
        ns1['_num_neurons'] = len(active)
        ns2['_array_v'] = v[inactive]
        ns2['_array_ge'] = ge[inactive]
        ns2['_array_gi'] = gi[inactive]
        ns2['_num_neurons'] = len(inactive)
        if len(active):
            su1.update()
            v[active] = ns1['_array_v']
            ge[active] = ns1['_array_ge']
            gi[active] = ns1['_array_gi']
        if len(inactive):
            su2.update()
            v[inactive] = ns2['_array_v']
            ge[inactive] = ns2['_array_ge']
            gi[inactive] = ns2['_array_gi']
        ns1['_array_v'] = v
        ns1['_array_ge'] = ge
        ns1['_array_gi'] = gi
    net = Network(P1, msu)
    reinitP(P1)
    net.run(1*ms)
    start = time.time()
    net.run(duration)
    end = time.time()
    return (end-start, sum(P1.v_))

print 'N:', N
print 'duration:', duration
print 'Brian 1.x:', brian_1x()
print 'Modify equations:', modify_equations()
print 'Multiple updaters:', multiple_updaters()

#M = SpikeMonitor(P)
#
#run(1000*ms)
#print M.num_spikes*1.0/len(P)
#
#plot(M.t, M.i, '.k')
#show()
