from ...codeobject import CodeObject
from ..templates import LanguageTemplater

#    def code_object(self, code, namespace, specifiers):
#        # TODO: This should maybe go somewhere else
#        namespace['logical_not'] = np.logical_not
#        return PythonCodeObject(code, namespace, specifiers,
#                                self.compile_methods(namespace))

class NumpyCodeObject(CodeObject):
    templater = LanguageTemplater(os.path.join(os.path.split(__file__)[0],
                                               'templates'))

    def compile(self):
        super(NumpyCodeObject, self).compile()
        self.compiled_code = compile(self.code, '(string)', 'exec')

    def run(self):
        exec self.compiled_code in self.namespace
        # output variables should land in the variable name _return_values
        if '_return_values' in self.namespace:
            return self.namespace['_return_values']
