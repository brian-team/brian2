import time
import warnings

from numpy import float64
import pylab

from brian2.groups.neurongroup import NeuronGroup

from brian2.codegen.specifiers import (Function, Value, ArrayVariable,
                                       Subexpression, Index)
from brian2.codegen.translation import translate, make_statements
from brian2.codegen.languages import (CPPLanguage, PythonLanguage, CUDALanguage,
                                      NumexprPythonLanguage)
from brian2.codegen.templating import apply_code_template
from brian2.utils.stringtools import deindent
from brian2.units import *
from brian2.units.stdunits import *

from codeprint import codeprint

test_compile = True
do_plot = True

tau = 10 * ms # external variable
eqs = '''   
    dV/dt = (-V + I + J)/tau : 1
    dI/dt = -I/tau : 1
    J = V * 0.1 : 1
    '''
from brian2.stateupdaters.integration import euler, rk2, rk4
G = NeuronGroup(1, model=eqs, method=rk4)

intermediate = make_statements(G.abstract_code, G.specifiers, float64)

print 'EQUATIONS:'
print eqs
print 'ABSTRACT CODE:'
print G.abstract_code
print 'INTERMEDIATE STATEMENTS:'
print
for stmt in intermediate:
    print stmt
print

def getlang(cls, *args, **kwds):
    try:
        return cls(*args, **kwds)
    except Exception as e:
        warnings.warn("Couldn't load language "+cls.__name__+'\n'+str(e))
        return None

languages = [lang for lang in [
    getlang(PythonLanguage),
    getlang(NumexprPythonLanguage),
    getlang(CPPLanguage, extra_compile_args=['-O3', '-ffast-math', '-march=native'], restrict='__restrict__', flush_denormals=True),
    #getlang(CUDALanguage),
    ] if lang is not None]

codestrings = {}

for lang in languages:
    innercode = translate(G.abstract_code, G.specifiers, float64, lang)
    code = lang.apply_template(innercode, lang.template_state_update())
    codestrings[lang] = code
    print lang.__class__.__name__
    print '='*len(lang.__class__.__name__)
    codeprint(code)

if not test_compile:
    exit()

N = 100000
Nshow = 10
tshow = 100
ttest = 1000
dt = 0.001
_array_V = pylab.rand(N)
_array_I = pylab.rand(N)
_array_tau = pylab.ones(N)*30*0.001

print
print 'Timings'
print '======='
print
print 'Num neurons =', N
print 'Duration =', ttest*0.001, 's'
print

Mall = {}
pylab.subplot(211)
pylab.title('All languages, some neurons')
for lang in languages:
    if lang not in codestrings:
        continue
    namespace = {
        '_array_V': _array_V.copy(),
        '_array_I': _array_I.copy(),
        '_array_tau': _array_tau.copy(),
        '_num_neurons': N,
        }
    code = codestrings[lang]
    try:
        codeobj = lang.code_object(code, G.specifiers)
    except NotImplementedError:
        print lang.__class__.__name__+':', 'not implemented'
        continue
    codeobj.compile(namespace)
    M = []
    T = []
    Mc = []
    for t in pylab.arange(100)*dt:
        M.append(namespace['_array_V'].copy()[:Nshow])
        Mc.append(namespace['_array_V'][0])
        T.append(t)
        codeobj(t=t, dt=dt)
    Mall[lang.__class__.__name__] = Mc
    pylab.plot(T, M)
    start = time.time()
    for i in xrange(ttest):
        # use this to cope with slow performance due to denormals
        if i%10000==0:
            namespace['_array_V'][:] = 1
        codeobj(t=t, dt=dt)
    end = time.time()
    print lang.__class__.__name__+':', end-start
pylab.subplot(212)
pylab.title('Comparison of single neuron')
for k, v in Mall.items():
    pylab.plot(T, v, label=k)
pylab.legend(loc='upper right')
if do_plot:
    pylab.show()
