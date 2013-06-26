'''
There's some hacky stuff in here to get around the fact that I don't deal
with specifiers properly.
'''
from numpy import float64, zeros, array, arange
from numpy.random import rand
from brian2.core.specifiers import (Value, ArrayVariable, 
                                    Subexpression, Index)
from brian2.codegen.translation import translate, make_statements
from brian2.codegen.languages import CPPLanguage, PythonLanguage
from brian2.utils.stringtools import deindent
from brian2 import ms
from brian2.memory.dynamicarray import DynamicArray1D

abstract = '''
_cond = _rand(_num_target_neurons)<0.1
'''

specifiers = {
    }

indices = {
    }

ns = {
    '_num_source_neurons': 20,
    '_num_target_neurons': 20,
    '_presynaptic': DynamicArray1D(0, dtype=int, use_numpy_resize=True),
    '_postsynaptic': DynamicArray1D(0, dtype=int, use_numpy_resize=True),
    'arange': arange,
    }

codeobjs = {}

langs = [PythonLanguage(),
         CPPLanguage(),
         ]

for lang in langs:
    innercode, kwds = translate(abstract, specifiers, {}, float64, lang, indices)
    code = lang.templater.synapses_create(innercode, **kwds)
    print lang.__class__.__name__
    print '='*len(lang.__class__.__name__)
    print code
    codeobj = lang.create_codeobj('synapses_create', abstract, ns, {}, lang.templater.synapses_create)
    codeobjs[lang.language_id] = codeobj

for lang in langs:
    s = lang.__class__.__name__+" (results)"
    print s+'\n'+'='*len(s)
    codeobj = codeobjs[lang.language_id]
    ns['_presynaptic'].shrink(0)
    ns['_postsynaptic'].shrink(0)
    if lang.language_id=='python':
        codeobj.namespace['_rand'] = rand
    codeobj()
    print '    _presynaptic =', ns['_presynaptic']
    print '    _postsynaptic =', ns['_postsynaptic']
    print '    fraction =', len(ns['_presynaptic'])/float(ns['_num_source_neurons']*ns['_num_target_neurons'])
