# -*- coding:utf-8 -*-
"""
Created on 17 Jul 2013

@author: dan
"""

__all__ = ['WeaveCodeObject']

class WeaveCodeObject(CodeObject):
    '''
    C++ code object
    
    The ``code`` should be a `~brian2.codegen.languages.templates.MultiTemplate`
    object with two macros defined, ``main`` (for the main loop code) and
    ``support_code`` for any support code (e.g. function definitions).
    '''
    templater = LanguageTemplater(os.path.join(os.path.split(__file__)[0],
                                               'templates'))
    def __init__(self, code, namespace, specifiers, compile_methods=[],
                 compiler='gcc', extra_compile_args=['-O3']):
        super(CPPCodeObject, self).__init__(code,
                                            namespace,
                                            specifiers,
                                            compile_methods=compile_methods)
        self.compiler = compiler
        self.extra_compile_args = extra_compile_args

    def run(self):
        return weave.inline(self.code.main, self.namespace.keys(),
                            local_dict=self.namespace,
                            support_code=self.code.support_code,
                            compiler=self.compiler,
                            extra_compile_args=self.extra_compile_args)
#
#    def code_object(self, code, namespace, specifiers):
#        return CPPCodeObject(code,
#                             namespace,
#                             specifiers,
#                             compile_methods=self.compile_methods(namespace),
#                             compiler=self.compiler,
#                             extra_compile_args=self.extra_compile_args)
