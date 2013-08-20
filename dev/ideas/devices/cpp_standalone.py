from numpy import *
from brian2 import *
from brian2.utils.stringtools import *
from brian2.codegen.languages.cpp_lang import *
from brian2.devices.cpp_standalone import *
from brian2.devices.cpp_standalone.codeobject import CPPStandaloneCodeObject
from brian2.core.variables import *
import os
from collections import defaultdict

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
vars = G.variables
arrays = {}
for k, v in vars.items():
    if isinstance(v, ArrayVariable):
        k = '_array_%s_%s' % (G.name, k)
        arrays[v] = (k, c_data_type(v.dtype), len(v.value))

# Generate data for non-constant values
code_object_defs = defaultdict(list)
for codeobj in code_objects:
    for k, v in codeobj.nonconstant_values:
        if k=='t' or k=='_spikes' or k=='_num_spikes':
            pass
        elif v.im_class is ArrayVariable:
            # find the corresponding array
            arr = v.im_self
            arr_k, arr_dtype, arr_N = arrays[arr]
            val = v()
            if isinstance(val, int):
                code_object_defs[codeobj].append('int %s = %s;' % (k, arr_N))
            elif k=='_spikespace':
                code_object_defs[codeobj].append('%s *%s = %s;' % (arr_dtype, k, arr_k))
            elif isinstance(val, ndarray):
                pass
                #code_object_defs[codeobj].append('%s *%s = %s;' % (arr_dtype, k, arr_k))
            else:
                raise ValueError("Unknown")
        else:
            raise ValueError("Unknown")

def constants(ns):
    return dict((k, v) for k, v in ns.iteritems() if isinstance(v, (int, float)))

def freeze(code, ns):
    # this is a bit of a hack, it should be passed to the template somehow
    for k, v in ns.items():
        if isinstance(v, float):
            code = ('const double %s = %s;\n' % (k, repr(v)))+code
        elif isinstance(v, int):
            code = ('const int %s = %s;\n' % (k, repr(v)))+code
    return code

if not os.path.exists('output'):
    os.mkdir('output')

main_tmp = CPPStandaloneCodeObject.templater.main(None,
                                                  code_objects=code_objects,
                                                  num_steps=1000,
                                                  dt=float(defaultclock.dt),
                                                  )
open('output/main.cpp', 'w').write(main_tmp)

arr_tmp = CPPStandaloneCodeObject.templater.arrays(None, array_specs=arrays.values())
open('output/arrays.cpp', 'w').write(arr_tmp.cpp_file)
open('output/arrays.h', 'w').write(arr_tmp.h_file)

for codeobj in code_objects:
    ns = codeobj.namespace
    # TODO: fix this hack
    # Surprise, surprise. Using global variables like this in C++ doesn't work. That's OK, this was
    # only ever a temporary hack anyway. You can manually modify the codeobject.cpp files to move
    # these definitions into the function, and then it works! Hurrah!
    code = freeze(codeobj.code.cpp_file, ns)
    code = '\n'.join(code_object_defs[codeobj])+code
    code = '#include "arrays.h"\n'+code
    
    open('output/'+codeobj.name+'.cpp', 'w').write(code)
    open('output/'+codeobj.name+'.h', 'w').write(codeobj.code.h_file)
