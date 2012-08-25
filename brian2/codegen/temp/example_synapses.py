from numpy import float64
from brian2.codegen.specifiers import (Function, Value, ArrayVariable,
                                       Subexpression, Index)
from brian2.codegen.translation import translate, make_statements
from brian2.codegen.languages import (CLanguage, PythonLanguage, CUDALanguage,
                                      NumexprPythonLanguage)
from brian2.codegen.templating import apply_code_template
from brian2.utils.stringtools import deindent

abstract = '''
V += w
'''

specifiers = {
    'V':ArrayVariable('_array_V', '_postsynaptic_idx', float64),
    'w':ArrayVariable('_array_w', '_synapse_idx', float64),
    '_spiking_synapse_idx':Index(),
    '_postsynaptic_idx':Index(all=False),
    '_synapse_idx':Index(all=False),
    '_presynaptic_idx':Index(all=False),
    }

intermediate = make_statements(abstract, specifiers, float64)

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
        ]:
    innercode = translate(abstract, specifiers, float64, lang)
    code = apply_code_template(innercode, deindent(lang.template_synapses()))
    print lang.__class__.__name__
    print '='*len(lang.__class__.__name__)
    print code
