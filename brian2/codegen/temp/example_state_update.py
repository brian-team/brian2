from numpy import float64
from brian2.codegen.specifiers import (Function, Value, ArrayVariable,
                                       Subexpression, Index)
from brian2.codegen.translation import translate, make_statements
from brian2.codegen.languages import (CLanguage, PythonLanguage, CUDALanguage,
                                      NumexprPythonLanguage)
from brian2.codegen.templating import apply_code_template
from brian2.utils.stringtools import deindent
import pylab
import time

test_compile = True

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
    'V':ArrayVariable('_array_V', '_neuron_idx', float64),
    'tau':ArrayVariable('_array_tau', '_neuron_idx', float64),
    'x':Subexpression('-V/tau'),
    'dt':Value(float64),
    '_neuron_idx':Index(all=True),
    }

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

languages = [
    PythonLanguage(),
    NumexprPythonLanguage(),
    CLanguage(extra_compile_args=['-O3', '-ffast-math', '-march=native'], restrict='__restrict__'),
    CUDALanguage(),
    ]

codestrings = {}

for lang in languages:
    innercode = translate(abstract, specifiers, float64, lang)
    #code = apply_code_template(innercode, deindent(lang.template_state_update()))
    code = lang.apply_template(innercode, lang.template_state_update())
    codestrings[lang] = code
    print lang.__class__.__name__
    print '='*len(lang.__class__.__name__)
    print code

if not test_compile:
    exit()

N = 10000
Nshow = 10
tshow = 100
ttest = 10000
dt = 0.001
_array_V = pylab.rand(N)
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
    namespace = {
        '_array_V': _array_V.copy(),
        '_array_tau': _array_tau.copy(),
        '_num_neurons': N,
        }
    code = codestrings[lang]
    try:
        codeobj = lang.code_object(code)
    except NotImplementedError:
        print lang.__class__.__name__+':', 'not implemented'
        continue
    codeobj.compile(namespace)
    M = []
    T = []
    Mc = []
    for t in pylab.arange(100)*dt:
        codeobj(t=t, dt=dt)
        M.append(namespace['_array_V'].copy()[:Nshow])
        Mc.append(namespace['_array_V'][0])
        T.append(t)
    Mall[lang.__class__.__name__] = Mc
    pylab.plot(T, M)
    start = time.time()
    for i in xrange(ttest):
        codeobj(t=t, dt=dt)
    end = time.time()
    print lang.__class__.__name__+':', end-start
pylab.subplot(212)
pylab.title('Comparison of single neuron')
for k, v in Mall.items():
    pylab.plot(T, v, label=k)
pylab.legend(loc='upper right')
pylab.show()
