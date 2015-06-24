from brian2.codegen.runtime.cython_rt.extension_manager import cython_extension_manager

code = '''
def f(ns):
    #cdef int n = <int> ns['n']
    cdef int n = ns['n']
    print n
'''

ns = {
    'n':3,
    }

mod = cython_extension_manager.create_extension(code)
mod.f(ns)