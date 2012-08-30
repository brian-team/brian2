from numpy import float64
from brian2.codegen.specifiers import (Function, Value, ArrayVariable,
                                       Subexpression, Index)
from brian2.codegen.translation import translate, make_statements
from brian2.codegen.languages import (CPPLanguage, PythonLanguage, CUDALanguage,
                                      NumexprPythonLanguage)
from brian2.codegen.templating import apply_code_template
from brian2.codegen.functions import make_user_function
from brian2.utils.stringtools import deindent
import pylab
from pylab import sin
from codeprint import codeprint

@make_user_function(codes={
    'cpp':{
        'support_code':"""
            #include<math.h>
            inline double usersin(double x)
            {
                return sin(x);
            }
            """,
        'hashdefine_code':'',
        },
    }, namespace={})
def usersin(x):
    return sin(x)

abstract = '''
I = usersin(2*pi*f*t)
'''

specifiers = {
    'I':ArrayVariable('_array_I', '_neuron_idx', float64),
    'f':ArrayVariable('_array_f', '_neuron_idx', float64),
    'pi':Value(float64),
    'usersin':usersin,
    '_neuron_idx':Index(all=True),
    }

languages = [
    PythonLanguage(),
    CPPLanguage(),
    ]

dt = 0.001
N = 10
I = pylab.zeros(N)
f = 1.0+pylab.arange(N)

for lang in languages:
    namespace = {
        '_array_I': I.copy(),
        '_array_f': f.copy(),
        'pi': pylab.pi,
        '_num_neurons': N,
        }
    innercode = translate(abstract, specifiers, float64, lang)
    print '*********', lang.__class__.__name__, '*********'
    print '========= INNER CODE ========='
    codeprint(innercode)
    print '=========== TEMPLATE ========='
    tmp = lang.template_state_update()
    codeprint(tmp)
    code = lang.apply_template(innercode, lang.template_state_update())
    print '=========== CODE ============='
    codeprint(code)
    codeobj = lang.code_object(code, specifiers)
    codeobj.compile(namespace)
    T = []
    M = []
    for t in pylab.arange(100)*dt:
        t = float(t)
        M.append(namespace['_array_I'].copy())
        T.append(t)
        codeobj(t=t)
    pylab.plot(T, M)
pylab.show()
