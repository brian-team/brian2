from numpy import float64
from brian2.codegen.specifiers import (Function, Value, ArrayVariable,
                                       Subexpression, Index)
from brian2.codegen.translation import translate, make_statements
from brian2.codegen.languages import CLanguage, PythonLanguage, CUDALanguage
from brian2.codegen.templating import apply_code_template
from brian2.utils.stringtools import deindent

# we don't actually use these, but this is what we would start from
reset = '''
V = Vr
'''

abstract = '''
V = Vr
'''

specifiers = {
    'V':ArrayVariable('_array_V', '_neuron_idx', float64),
    'Vr':ArrayVariable('_array_Vr', '_neuron_idx', float64),
    '_neuron_idx':Index(all=False),
    }

intermediate = make_statements(abstract, specifiers, float64)

print 'RESET:'
print reset
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
    code = apply_code_template(innercode, lang.template_reset())
    print lang.__class__.__name__
    print '='*len(lang.__class__.__name__)
    print code
