#!/usr/bin/env python
'''
Input-Frequency curve of a IF model.
Network: 1000 unconnected integrate-and-fire neurons (leaky IF)
with an input parameter v0.
The input is set differently for each neuron.
'''
from brian2 import *
from brian2.core.variables import Variables, ArrayVariable, DynamicArrayVariable
from brian2.utils.stringtools import get_identifiers
import re
from brian2 import DEFAULT_FUNCTIONS, DEFAULT_CONSTANTS

tau = 10*ms

prefs.codegen.target = 'cython'

prefs.codegen.cpp.libraries += ['gsl', 'gslcblas']
prefs.codegen.cpp.headers += ['gsl/gsl_odeiv2.h']
prefs.codegen.cpp.include_dirs += ['/home/charlee/so   ftwarefolder/gsl-2.3/gsl/']

n = 10
duration = .1*second

eqs = '''
dv/dt = (v1 - v) / tau : volt (unless refractory)
v1 = v0*multi : volt
multi = 2 : 1
v0 : volt
'''

class GSLStateUpdater(StateUpdateMethod):
    def __call__(self, equations, variables=None):
        # the approach is to write all variables so they can
        # be translated to GSL code (i.e. with indexing and pointers)
        diff_eqs = equations.get_substituted_expressions(variables)

        code = []
        count_statevariables = 0
        counter = {}
        defined = [x[0] for x in equations.eq_expressions]

        # differential equation variables (have to be changed to f[0] in
        # lhs and y[0] in rhs
        for name in equations.diff_eq_names:
            new_name = device.get_array_name(variables[name])
            code += ['{var_single} = _gsl_{var}_y{count}'.format(var_single=name, var=new_name, count=count_statevariables)]
            counter[name] = count_statevariables
            count_statevariables += 1

        # all the non-differential statevariables added to parameter struct
        # if array than take array name (and later save to params as pointer)
        print equations, equations.names
        for name in equations.names:
            if name in defined:
                continue
            var = variables[name]
            if isinstance(var, ArrayVariable):
                new_name = device.get_array_name(var)
                code += ['{var_single} = _gsl_{var}_p'.format(var_single=name, var=new_name)]
            else:
                code += ['{var} = _gsl_{var}_p'.format(var=name)]

        for name in equations.identifiers:
            if name in DEFAULT_FUNCTIONS or name in DEFAULT_CONSTANTS:
                pass
            else:
                code += ['{var} = _gsl_{var}_p'.format(var=name)]

        for name, expr in diff_eqs:
            new_name = device.get_array_name(variables[name])

            code += ['_gsl_{var}_f{count} = {expr}'.format(var=new_name,
                                                           expr=expr,
                                                           count=counter[name])]
        print code
        return ('\n').join(code)

group = NeuronGroup(n, eqs, threshold='v > 10*mV', reset='v = 0*mV',
                    refractory=5*ms, method=GSLStateUpdater())

group.v = 0*mV
group.v0 = '20*mV * i / (n-1)'

print group.v0

monitor = SpikeMonitor(group)
mon2 = StateMonitor(group, 'v', record=True)

run(duration)
print group.state_updater.codeobj.code
plot(group.v0/mV, monitor.count / duration)
xlabel('v0 (mV)')
ylabel('Firing rate (sp/s)')
show()
