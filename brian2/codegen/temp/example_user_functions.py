from numpy import float64
from brian2.codegen.specifiers import (Function, Value, ArrayVariable,
                                       Subexpression, Index)
from brian2.codegen.translation import translate, make_statements
from brian2.codegen.languages import (CPPLanguage, PythonLanguage, CUDALanguage,
                                      NumexprPythonLanguage)
from brian2.codegen.templating import apply_code_template
from brian2.codegen.functions import UserFunction
from brian2.utils.stringtools import deindent
import pylab

class TimedArray(UserFunction):
    def __init__(self, values, dt):
        self.values = values
        self.dt = dt
        self.len_values = len(values)
        
    def __call__(self, t):
        i = int(t+0.5)
        if i<0: i = 0
        if i>=self.len_values: i = self.len_values-1
        return self.values[i]
        
    def code_cpp(self, language, var):
        support_code = '''
        inline double _func_%VAR%(const double t, const double _dt_%VAR%, const int _len_%VAR%, const double* _values_%VAR%)
        {
            int i = (int)(t/_dt_%VAR%+0.5); // rounds to nearest int for positive values
            if(i<0) i = 0;
            if(i>=_len_%VAR%) i = _len_%VAR%-1;
            return _values_%VAR%[i];
        }
        '''.replace('%VAR%', var)
        hashdefine_code = '''
        #define %VAR%(t) _func_%VAR%(t, _dt_%VAR%, _len_%VAR%, _values_%VAR%)
        '''.replace('%VAR%', var)
        return {'support_code': support_code,
                'hashdefine_code': hashdefine_code,
                }
    
    def on_compile_cpp(self, namespace, language, var):
        namespace['_len_'+var] = self.len_values
        namespace['_values_'+var] = self.values
        namespace['_dt_'+var] = self.dt

dt = 0.001
values = pylab.sin(pylab.arange(100)*dt*50*pylab.pi)

f = TimedArray(values, dt)

abstract = '''
I = f(t)*J
'''

specifiers = {
    'I':ArrayVariable('_array_I', '_neuron_idx', float64),
    'J':ArrayVariable('_array_J', '_neuron_idx', float64),
    'f':f,
    'dt':Value(float64),
    '_neuron_idx':Index(all=True),
    }

languages = [
    PythonLanguage(),
    CPPLanguage(),
    ]

N = 10
I = pylab.zeros(N)
J = pylab.arange(N)*1.0

for lang in languages:
    namespace = {
        '_array_I': I.copy(),
        '_array_J': J.copy(),
        '_num_neurons': N,
        'dt': dt,
        }
    innercode = translate(abstract, specifiers, float64, lang)
    print
    print '========= INNER CODE ========='
    print
    if isinstance(innercode, str):
        print innercode
    else:
        for k, v in innercode.items():
            print k+':'
            print v
    print
    print '=========== TEMPLATE ========='
    print
    tmp = lang.template_state_update()
    if isinstance(tmp, str):
        print tmp
    else:
        for k, v in tmp.items():
            print k+':'
            print v
    code = lang.apply_template(innercode, lang.template_state_update())
    print
    print '============ CODE =============='
    print
    if isinstance(code, str):
        print code
    else:
        for k, v in code.items():
            print k+':'
            print v
    codeobj = lang.code_object(code, specifiers)
    codeobj.compile(namespace)
    T = []
    M = []
    for t in pylab.arange(100)*dt:
        M.append(namespace['_array_I'].copy())
        T.append(t)
        codeobj(t=t)
    pylab.plot(T, M)
pylab.show()
