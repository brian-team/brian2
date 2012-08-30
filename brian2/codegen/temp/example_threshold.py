from numpy import float64
from brian2.codegen.specifiers import (Function, Value, ArrayVariable,
                                       OutputVariable, Subexpression, Index)
from brian2.codegen.translation import translate, make_statements
from brian2.codegen.languages import CPPLanguage, PythonLanguage, CUDALanguage
from brian2.codegen.templating import apply_code_template
from brian2.utils.stringtools import deindent
from codeprint import codeprint

# we don't actually use these, but this is what we would start from
threshold = 'V>Vt'

abstract = '''
_cond = V>Vt
'''

specifiers = {
    'V':ArrayVariable('_array_V', '_neuron_idx', float64),
    'Vt':ArrayVariable('_array_Vt', '_neuron_idx', float64),
    '_cond':OutputVariable(bool),
    '_neuron_idx':Index(all=True),
    }

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
        CPPLanguage(),
        CUDALanguage(),
        ]:
    innercode = translate(abstract, specifiers, float64, lang)
    code = lang.apply_template(innercode, lang.template_threshold())
    print lang.__class__.__name__
    print '='*len(lang.__class__.__name__)
    codeprint(code)
