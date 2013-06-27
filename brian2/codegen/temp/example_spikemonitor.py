from numpy import float64, zeros, array
from brian2.core.specifiers import (Value, ArrayVariable,
                                    Subexpression, Index)
from brian2.codegen.translation import translate, make_statements
from brian2.codegen.languages import CPPLanguage, PythonLanguage
from brian2.utils.stringtools import deindent
from brian2 import ms
from brian2.memory.dynamicarray import DynamicArray1D

abstract = '''
'''

specifiers = {
    }

indices = {
    }

ns = {
    '_spikes':zeros(0, dtype=int),
    't_arr': DynamicArray1D(0, dtype=float, use_numpy_resize=True),
    'i_arr': DynamicArray1D(0, dtype=int, use_numpy_resize=True),
    }

codeobjs = {}

langs = [PythonLanguage(), CPPLanguage()]

for lang in langs:
    innercode, kwds = translate(abstract, specifiers, {}, float64, lang, indices)
    code = lang.templater.monitor(innercode, **kwds)
    print lang.__class__.__name__
    print '='*len(lang.__class__.__name__)
    print code
    codeobj = lang.create_codeobj('monitor', '', ns, {}, lang.templater.monitor)
    codeobjs[lang.language_id] = codeobj

for lang in langs:
    s = lang.__class__.__name__+" (results)"
    print s+'\n'+'='*len(s)
    codeobj = codeobjs[lang.language_id]
    ns['t_arr'].shrink(0)
    ns['i_arr'].shrink(0)
    codeobj(t=1*ms, _spikes=array([1,2], dtype=int), _num_spikes=2)
    codeobj(t=2*ms, _spikes=array([0,2], dtype=int), _num_spikes=2)
    print '    t_arr =', ns['t_arr']
    print '    i_arr =', ns['i_arr']
