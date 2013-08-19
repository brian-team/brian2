from numpy import *
from brian2 import *
from brian2.utils.stringtools import *
from brian2.codegen.languages.cpp_lang import *
from brian2.devices.cpp_standalone import *
from brian2.devices.cpp_standalone.codeobject import CPPStandaloneCodeObject
import os

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
                )
# Run the network for 0 seconds to generate the code
net = Network(G)
net.run(0*second)

# Extract all the CodeObjects
# Here we hack it directly, as there are more general issues to solve before we can do this automatically
code_objects = [G.state_updater.codeobj,
                G.resetter.codeobj,
                G.thresholder.codeobj,
                ]

# Extract the array information
ns = G.state_updater.codeobj.namespace
arrays = []
# Freeze all constants
for k, v in ns.items():
#    if isinstance(v, float):
#        code = ('const double %s = %s;\n' % (k, repr(v)))+code
#    elif isinstance(v, int):
#        code = ('const int %s = %s;\n' % (k, repr(v)))+code
    if isinstance(v, ndarray):
        if k.startswith('_array'):
            dtype_spec = c_data_type(v.dtype)
            arrays.append((k, dtype_spec, N))

if not os.path.exists('output'):
    os.mkdir('output')

arr_tmp = CPPStandaloneCodeObject.templater.arrays(None, array_specs=arrays)
open('output/arrays.cpp', 'w').write(arr_tmp.cpp_file)
open('output/arrays.h', 'w').write(arr_tmp.h_file)

for codeobj in code_objects:
    open('output/'+codeobj.name+'.cpp', 'w').write(codeobj.code.cpp_file)
    open('output/'+codeobj.name+'.h', 'w').write(codeobj.code.h_file)
