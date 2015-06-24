from pylab import *
import cython
import time, timeit
from brian2.codegen.runtime.cython_rt.modified_inline import modified_cython_inline
import numpy
from scipy import weave
import numexpr
import theano
from theano import tensor as tt

tau = 20 * 0.001
N = 1000000
b = 1.2 # constant current mean, the modulation varies
freq = 10.0
t = 0.0
dt = 0.0001

_array_neurongroup_a = a = linspace(.05, 0.75, N)
_array_neurongroup_v = v = rand(N)

ns = {'_array_neurongroup_a': a, '_array_neurongroup_v': v,
      '_N': N,
      'dt': dt, 't': t, 'tau': tau, 'b': b, 'freq': freq,# 'sin': numpy.sin,
      'pi': pi,
      }

code = '''
cdef int _idx
cdef int _vectorisation_idx
cdef int N = <int>_N
cdef double a, v, _v
#cdef double [:] _cy_array_neurongroup_a = _array_neurongroup_a
#cdef double [:] _cy_array_neurongroup_v = _array_neurongroup_v
cdef double* _cy_array_neurongroup_a = &(_array_neurongroup_a[0])
cdef double* _cy_array_neurongroup_v = &(_array_neurongroup_v[0])
for _idx in range(N):
    _vectorisation_idx = _idx
    a = _cy_array_neurongroup_a[_idx]
    v = _cy_array_neurongroup_v[_idx]
    _v = a*sin(2.0*freq*pi*t) + b + v*exp(-dt/tau) + (-a*sin(2.0*freq*pi*t) - b)*exp(-dt/tau)
    #_v = a*b+0.0001*sin(v)
    #_v = a*b+0.0001*v
    v = _v
    _cy_array_neurongroup_v[_idx] = v
'''
def timefunc_cython_inline():
    cython.inline(code, locals=ns)

f_mod, f_arg_list = modified_cython_inline(code, locals=ns, globals={})
def timefunc_cython_modified_inline():
    f_mod.__invoke(*f_arg_list)
    #modified_cython_inline(code, locals=ns)

def timefunc_python():
    for _idx in xrange(N):
        _vectorisation_idx = _idx
        a = _array_neurongroup_a[_idx]
        v = _array_neurongroup_v[_idx]
        _v = a*sin(2.0*freq*pi*t) + b + v*exp(-dt/tau) + (-a*sin(2.0*freq*pi*t) - b)*exp(-dt/tau)
        v = _v
        _array_neurongroup_v[_idx] = v
        
def timefunc_numpy():
    _v = a*sin(2.0*freq*pi*t) + b + v*exp(-dt/tau) + (-a*sin(2.0*freq*pi*t) - b)*exp(-dt/tau)
    v[:] = _v

def timefunc_numpy_smart():
    _sin_term = sin(2.0*freq*pi*t)
    _exp_term = exp(-dt/tau)
    _a_term = (_sin_term-_sin_term*_exp_term)
    _v = v
    _v *= _exp_term
    _v += a*_a_term
    _v += -b*_exp_term + b
    
def timefunc_numpy_blocked():
    ext = exp(-dt/tau)
    sit = sin(2.0*freq*pi*t)
    bs = 20000
    for i in xrange(0, N, bs):
        ab = a[i:i+bs]
        vb = v[i:i+bs]
        absit = ab*sit + b
        vb *= ext
        vb += absit
        vb -= absit*ext

def timefunc_numexpr():
    v[:] = numexpr.evaluate('a*sin(2.0*freq*pi*t) + b + v*exp(-dt/tau) + (-a*sin(2.0*freq*pi*t) - b)*exp(-dt/tau)')

def timefunc_numexpr_smart():
    _sin_term = sin(2.0*freq*pi*t)
    _exp_term = exp(-dt/tau)
    _a_term = (_sin_term-_sin_term*_exp_term)
    _const_term = -b*_exp_term + b
    #v[:] = numexpr.evaluate('a*_a_term+v*_exp_term+_const_term')
    numexpr.evaluate('a*_a_term+v*_exp_term+_const_term', out=v)
    
def timefunc_weave(*args):
    code = '''
// %s
int N = _N;
for(int _idx=0; _idx<N; _idx++)
{
double a = _array_neurongroup_a[_idx];
double v = _array_neurongroup_v[_idx];
double _v = a*sin(2.0*freq*pi*t) + b + v*exp(-dt/tau) + (-a*sin(2.0*freq*pi*t) - b)*exp(-dt/tau);
v = _v;
_array_neurongroup_v[_idx] = v;
}
''' % str(args)
    weave.inline(code, ns.keys(), ns, compiler='gcc', extra_compile_args=list(args))
    
def timefunc_weave_slow():
    timefunc_weave('-O3', '-march=native')

def timefunc_weave_fast():
    timefunc_weave('-O3', '-march=native', '-ffast-math')
    
    
def get_theano_func():
    a = tt.dvector('a')
    v = tt.dvector('v')
    freq = tt.dscalar('freq')
    t = tt.dscalar('t')
    dt = tt.dscalar('dt')
    tau = tt.dscalar('tau')
    return theano.function([a, v, freq, t, dt, tau],
                           a*tt.sin(2.0*freq*pi*t) + b + v*tt.exp(-dt/tau) + (-a*tt.sin(2.0*freq*pi*t) - b)*tt.exp(-dt/tau))
# return theano.function([a, v],
# a*tt.sin(2.0*freq*pi*t) + b + v*tt.exp(-dt/tau) + (-a*tt.sin(2.0*freq*pi*t) - b)*tt.exp(-dt/tau))

theano.config.gcc.cxxflags = '-O3 -ffast-math'
theano_func = get_theano_func()
#print theano.pp(theano_func.maker.fgraph.outputs[0])
#print
#theano.printing.debugprint(theano_func.maker.fgraph.outputs[0])
#theano.printing.pydotprint(theano_func, 'func.png')
#exit()
    
def timefunc_theano():
    v[:] = theano_func(a, v, freq, t, dt, tau)

def dotimeit(f):
    v[:] = 1
    f()
    print '%s: %.2f' % (f.__name__.replace('timefunc_', ''),
                        timeit.timeit(f.__name__+'()', setup='from __main__ import '+f.__name__, number=100))

def check_values(f):
    v[:] = 1
    v[:5] = linspace(0, 1, 5)
    f()
    print '%s: %s' % (f.__name__.replace('timefunc_', ''), v[:5])

if __name__=='__main__':
    funcs = [#timefunc_cython_inline,
             timefunc_cython_modified_inline,
             timefunc_numpy,
             timefunc_numpy_smart,
             timefunc_numpy_blocked,
             timefunc_numexpr,
             timefunc_numexpr_smart,
             timefunc_weave_slow,
             timefunc_weave_fast,
             timefunc_theano,
             ]
    if 1:
        print 'Values'
        print '======'
        for f in funcs:
            check_values(f)
        print
    if 1:
        print 'Times'
        print '====='
        for f in funcs:
            dotimeit(f)
