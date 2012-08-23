from numpy import float64
from brian2.codegen.specifiers import (Function, Value, ArrayVariable,
                                       Subexpression, Index)
from brian2.codegen.translation import translate, make_statements
from brian2.codegen.languages import CLanguage, PythonLanguage, CUDALanguage
from brian2.codegen.templating import apply_code_template
from brian2.utils.stringtools import deindent

# we don't actually use these, but this is what we would start from
eqs = '''
dV/dt = x : volt
x = -V/tau : volt/second
tau : second
'''

abstract = '''
_tmp_V = x
V += _tmp_V*dt
'''

specifiers = {
    'V':ArrayVariable('_array_V', float64),
    'tau':ArrayVariable('_array_tau', float64),
    'x':Subexpression('-V/tau'),
    'dt':Value(float64),
    '_neuron_idx':Index(all=True),
    }

def template(lang):
    if isinstance(lang, CUDALanguage):
        return deindent('''
        __global__ stateupdate(int _num_neurons, double dt)
        {
            const int _neuron_idx = threadIdx.x+blockIdx.x*blockDim.x;
            if(_neuron_idx>=_num_neurons) return;
            %CODE%
        }
        ''')
    elif isinstance(lang, CLanguage):
        return deindent('''
        for(int _neuron_idx=0; _neuron_idx<_num_neurons; _neuron_idx++)
        {
            %CODE%
        }
        ''')
    elif isinstance(lang, PythonLanguage):
        return deindent('''
        %CODE%
        ''')

intermediate = make_statements(abstract, specifiers, float64)

print 'EQUATIONS:'
print eqs
print 'ABSTRACT CODE:'
print abstract
print 'INTERMEDIATE STATEMENTS:'
print
for stmt in intermediate:
    print stmt
print

for lang in [
        PythonLanguage(),
        CLanguage(),
        CUDALanguage(),
        ]:
    innercode = translate(abstract, specifiers, float64, lang)
    code = apply_code_template(innercode, template(lang))
    print lang.__class__.__name__
    print '='*len(lang.__class__.__name__)
    print code
