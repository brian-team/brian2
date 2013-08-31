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
N = 1000

##### Generate C++ code

# Use a NeuronGroup to fake the whole process
G = NeuronGroup(N, eqs, reset=reset, threshold=threshold, refractory=refractory, name='gp')
G2 = NeuronGroup(1, eqs, reset=reset, threshold=threshold, refractory=refractory, name='gp2')
# Run the network for 0 seconds to generate the code
net = Network(G, G2)
net.run(0*second)

# Extract all the CodeObjects
# Note that since we ran the Network object, these CodeObjects will be sorted into the right
# running order, assuming that there is only one clock
code_objects = []
for obj in net.objects:
    code_objects.extend(obj.code_objects)

# Extract the arrays information
vars = {}
for obj in net.objects:
    if hasattr(obj, 'variables'):
        for k, v in obj.variables.iteritems():
            vars[(obj, k)] = v
arrays = {}
for (obj, k), v in vars.items():
    if isinstance(v, ArrayVariable):
        k = '_array_%s_%s' % (obj.name, k)
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
                code_object_defs[id(codeobj)].append('int %s = %s;' % (k, arr_N))
            elif k=='_spikespace':
                code_object_defs[id(codeobj)].append('%s *%s = %s;' % (arr_dtype, k, arr_k))
            elif isinstance(val, ndarray):
                pass
            else:
                raise ValueError("Unknown")
        else:
            raise ValueError("Unknown")

def constants(ns):
    return dict((k, v) for k, v in ns.iteritems() if isinstance(v, (int, float)))

def freeze(code, ns):
    # this is a bit of a hack, it should be passed to the template somehow
    for k, v in ns.items():
        if isinstance(v, (int, float)):
            code = word_substitute(code, {k: repr(v)})
    return code

if not os.path.exists('output'):
    os.mkdir('output')

# The code_objects are passed in the right order to run them because they were
# sorted by the Network object. To support multiple clocks we'll need to be
# smarter about that.
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
    # TODO: fix these freeze/CONSTANTS hacks somehow - they work but not elegant. 
    code = freeze(codeobj.code.cpp_file, ns)
    code = code.replace('%CONSTANTS%', '\n'.join(code_object_defs[id(codeobj)]))
    code = '#include "arrays.h"\n'+code
    
    open('output/'+codeobj.name+'.cpp', 'w').write(code)
    open('output/'+codeobj.name+'.h', 'w').write(codeobj.code.h_file)

