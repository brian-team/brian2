from numpy import float64
from brian2.codegen.specifiers import (Function, Value, ArrayVariable,
                                       OutputVariable, Subexpression, Index)
from brian2.codegen.translation import translate, make_statements
from brian2.codegen.languages import CLanguage, PythonLanguage, CUDALanguage
from brian2.codegen.templating import apply_code_template
from brian2.utils.stringtools import deindent

# we don't actually use these, but this is what we would start from
threshold = 'V>Vt'

abstract = '''
_cond = V>Vt
'''

specifiers = {
    'V':ArrayVariable('_array_V', float64),
    'Vt':ArrayVariable('_array_Vt', float64),
    '_cond':OutputVariable(bool),
    '_neuron_idx':Index(all=True),
    }

def template(lang):
    if isinstance(lang, CUDALanguage):
        return deindent('''
        __global__ threshold(int _num_neurons)
        {
            const int _neuron_idx = threadIdx.x+blockIdx.x*blockDim.x;
            if(_neuron_idx>=_num_neurons) return;
            %CODE%
            _array_cond[_neuron_idx] = _cond;
        }
        ''')
    elif isinstance(lang, CLanguage):
        return deindent('''
        int _numspikes = 0;
        for(int _neuron_idx=0; _neuron_idx<_num_neurons; _neuron_idx++)
        {
            %CODE%
            if(_cond) {
                _spikes[_numspikes++] = _neuron_idx];
            }
        }
        ''')
    elif isinstance(lang, PythonLanguage):
        return deindent('''
        %CODE%
        return _cond.nonzero()[0]
        ''')

intermediate = make_statements(abstract, specifiers, float64)

print 'THRESHOLD:', threshold
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
