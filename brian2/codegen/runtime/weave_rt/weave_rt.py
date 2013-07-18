import os

from scipy import weave

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


class WeaveCodeObject(CodeObject):
    '''
    Weave code object
    
    The ``code`` should be a `~brian2.codegen.languages.templates.MultiTemplate`
    object with two macros defined, ``main`` (for the main loop code) and
    ``support_code`` for any support code (e.g. function definitions).
    '''
    templater = Templater(os.path.join(os.path.split(__file__)[0],
                                       'templates'))
    language = CPPLanguage()

    def __init__(self, code, namespace, specifiers):
        super(WeaveCodeObject, self).__init__(code, namespace, specifiers)
        self.compiler = brian_prefs['codegen.runtime.weave.compiler']
        self.extra_compile_args = brian_prefs['codegen.runtime.weave.extra_compile_args']

    def run(self):
        return weave.inline(self.code.main, self.namespace.keys(),
                            local_dict=self.namespace,
                            support_code=self.code.support_code,
                            compiler=self.compiler,
                            extra_compile_args=self.extra_compile_args)

runtime_targets['weave'] = WeaveCodeObject
