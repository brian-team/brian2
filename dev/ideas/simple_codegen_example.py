'''
TODO: modify this so to just generate snippets
'''
from numpy import *
from brian2 import *
from brian2.utils.stringtools import *
from brian2.codegen.generators.cpp_lang import *

##### Define the model
tau = 10*ms
eqs = '''
dV/dt = -V/tau : volt (unless-refractory)
'''
threshold = 'V>-50*mV'
reset = 'V=-60*mV'
refractory = 5*ms
groupname = 'gp'
N = 1000

##### Generate C++ code

# Use a NeuronGroup to fake the whole process
G = NeuronGroup(N, eqs, reset=reset, threshold=threshold,
                refractory=refractory, name=groupname,
                codeobj_class=WeaveCodeObject,
                )
# Run the network for 0 seconds to generate the code
net = Network(G)
net.run(0*second)
# Extract the necessary information
ns = G.state_updater.codeobj.namespace
code = deindent(G.state_updater.codeobj.code.main)
arrays = []
# Freeze all constants
for k, v in ns.items():
    if isinstance(v, float):
        code = ('const double %s = %s;\n' % (k, repr(v)))+code
    elif isinstance(v, int):
        code = ('const int %s = %s;\n' % (k, repr(v)))+code
    elif isinstance(v, ndarray):
        if k.startswith('_array'):
            dtype_spec = c_data_type(v.dtype)
            arrays.append((k, dtype_spec, N))

print '*********** DECLARATIONS **********'
# This is just an example of what you could do with declarations, generate your
# own code here...
for varname, dtype_spec, N in arrays:
    print '%s *%s = new %s [%s];' % (dtype_spec, varname, dtype_spec, N)

print '*********** MAIN LOOP *************'
print code
