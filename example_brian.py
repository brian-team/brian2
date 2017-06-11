#!/usr/bin/env python
'''
Input-Frequency curve of a IF model.
Network: 1000 unconnected integrate-and-fire neurons (leaky IF)
with an input parameter v0.
The input is set differently for each neuron.
'''
from brian2 import *
from brian2.core.variables import Variables, ArrayVariable, DynamicArrayVariable
from brian2.core.functions import Function
from brian2.utils.stringtools import get_identifiers
import re
from brian2 import DEFAULT_FUNCTIONS, DEFAULT_CONSTANTS

prefs.codegen.target = 'cython'

prefs.codegen.cpp.libraries += ['gsl', 'gslcblas']
prefs.codegen.cpp.headers += ['gsl/gsl_odeiv2.h']
prefs.codegen.cpp.include_dirs += ['/home/charlee/softwarefolder/gsl-2.3/gsl/']

n = 10
duration = .1*second

eqs = '''
dv/dt = (v1 - v) / tau : volt (unless refractory)
v1 = v0*multi : volt
multi = 2 : 1
v0 : volt
tau = 10*ms : second
'''

from brian2.units.stdunits import stdunits
class GSLStateUpdater(StateUpdateMethod):
    def __call__(self, equations, variables=None):
        # the approach is to write all variables so they can
        # be translated to GSL code (i.e. with indexing and pointers)
        diff_eqs = equations.get_substituted_expressions(variables)

        code = []
        count_statevariables = 0
        counter = {}
        defined = [x[0] for x in equations.eq_expressions] + ['t']

        for diff_name, expr in diff_eqs:
            new_diff_name = device.get_array_name(variables[diff_name])
            code += ['{var_single} = _gsl_{var}_y{count}'.format(var_single=diff_name, var=new_diff_name, count=count_statevariables)]
            counter[diff_name] = count_statevariables
            count_statevariables += 1

            for name in get_identifiers(str(expr)):
                var = variables[name]
                if isinstance(var, Function)\
                        or name in stdunits\
                        or name in defined:
                    continue

        code += ['_gsl_{var}_f{count} = {expr}'.format(var=new_diff_name,
                                                           expr=expr,
                                                           count=counter[name])]

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
