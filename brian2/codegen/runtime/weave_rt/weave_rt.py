import os
import numpy

try:
    from scipy import weave
    from scipy.weave.c_spec import num_to_c_types
except ImportError:
    # No weave for Python 3
    weave = None

from ...codeobject import CodeObject
from ...templates import Templater
from ...languages.cpp_lang import CPPLanguage
from ..targets import runtime_targets

from brian2.core.preferences import brian_prefs, BrianPreference

__all__ = ['WeaveCodeObject']

# Preferences
brian_prefs.register_preferences(
    'codegen.runtime.weave',
    'Weave runtime codegen preferences',
    compiler = BrianPreference(
        default='gcc',
        validator=lambda pref: pref=='gcc',
        docs='''
        Compiler to use for weave.
        ''',
        ),
    extra_compile_args = BrianPreference(
        default=['-w', '-O3', '-ffast-math'],
        docs='''
        Extra compile arguments to pass to compiler
        ''',
        ),
    )


def weave_data_type(dtype):
    '''
    Gives the C language specifier for numpy data types using weave. For example,
    ``numpy.int32`` maps to ``long`` in C.
    '''
    # this handles the case where int is specified, it will be int32 or int64
    # depending on platform
    if dtype is int:
        dtype = numpy.array([1]).dtype.type
    if dtype is float:
        dtype = numpy.array([1.]).dtype.type
        
    dtype = numpy.empty(0, dtype=dtype).dtype.char
        
    return num_to_c_types[dtype]
    

class WeaveCodeObject(CodeObject):
    '''
    Weave code object
    
    The ``code`` should be a `~brian2.codegen.languages.templates.MultiTemplate`
    object with two macros defined, ``main`` (for the main loop code) and
    ``support_code`` for any support code (e.g. function definitions).
    '''
    templater = Templater(os.path.join(os.path.split(__file__)[0],
                                       'templates'))
    language = CPPLanguage(c_data_type=weave_data_type)

    def __init__(self, code, namespace, variables, name='weave_code_object*'):
        super(WeaveCodeObject, self).__init__(code, namespace, variables, name=name)
        self.compiler = brian_prefs['codegen.runtime.weave.compiler']
        self.extra_compile_args = brian_prefs['codegen.runtime.weave.extra_compile_args']

    def run(self):
        return weave.inline(self.code.main, self.namespace.keys(),
                            local_dict=self.namespace,
                            support_code=self.code.support_code,
                            compiler=self.compiler,
                            extra_compile_args=self.extra_compile_args)

runtime_targets['weave'] = WeaveCodeObject
