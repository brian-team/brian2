'''
I'm showing how the CUBA script could be entirely turned to code.
'''
from brian import *

set_global_preferences(compile=True)

# Parameters and definitions: nothing special happens here
taum = 20 * ms
taue = 5 * ms
taui = 10 * ms
Vt = -50 * mV
Vr = -60 * mV
El = -49 * mV
we = (60 * 0.27 / 10) * mV # excitatory synaptic weight (voltage)
wi = (-20 * 4.5 / 10) * mV # inhibitory synaptic weight
eqs = '''
dv/dt  = (ge+gi-(v-El))/taum : volt
dge/dt = -ge/taue : volt
dgi/dt = -gi/taui : volt
'''

'''
Here a virtual neuron group is created. That is, the array of state variables
is not created. Code (possibly abstract code at this stage) is created for
state update, reset, threshold and an initialization code.
These are methods of a class.
Code is also output to create an instance.
All code is pushed to the stack of statements.
'''
P = NeuronGroup(4000, model=eqs, threshold='v>Vt', reset='v=Vr', refractory=5 * ms)

'''
Each of these statements produces code pushed to the stack.
'''
P.v = Vr
P.ge = 0 * mV
P.gi = 0 * mV

"Virtual subgroups are created (no code output)"
Pe = P.subgroup(3200)
Pi = P.subgroup(800)

"Code generated for two subclasses, and virtual Synapses created"
Se = Synapses(Pe, P, model = 'w : 1', pre = 'ge += we')
Si = Synapses(Pi, P, model = 'w : 1', pre = 'gi += wi')

'Each statement produces code'
Se[:,:]=0.02
Si[:,:]=0.02
Se.delay='rand()*ms'
Si.delay='rand()*ms'

'''
This is a statement that cannot produce code because the assignment
depends on an external array.
Instead, one should write:
P.v = 'Vr+rand()*(Vt-Vr)'
'''
P.v = Vr + rand(len(P)) * (Vt - Vr)

'''
Each of these monitors are classes with a target-specific implementation.
Each statement produces a code that creates an instance.
'''
# Record the number of spikes
Me = PopulationSpikeCounter(Pe)
Mi = PopulationSpikeCounter(Pi)
# A population rate monitor
M = PopulationRateMonitor(P)

'''
This produces the code for the main loop.
'''
run(1*second)

'''
What is needed now is saving the network or monitor information.
To be able to do this on the target, this should be done through methods
of monitors. For example:

Me.save('filename.dat')

Alternatively, saving could be done online:
M = PopulationRateMonitor(P,save='filename.txt')

Finally, we need an instruction to save or compile the code:
save_code('filename') # could be a folder if there are multiple files

Or even:
compile_code('filename')
'''
