import os
import numpy as np

from ...codeobject import CodeObject
from ...templates import Templater
from ...languages.numpy_lang import NumpyLanguage
from ..targets import runtime_targets

__all__ = ['NumpyCodeObject']

class NumpyCodeObject(CodeObject):
    '''
    Execute code using Numpy
    
    Default for Brian because it works on all platforms.
    '''
    templater = Templater(os.path.join(os.path.split(__file__)[0],
                                       'templates'))
    language = NumpyLanguage()

    def __init__(self, code, namespace, variables):
        # TODO: This should maybe go somewhere else
        namespace['logical_not'] = np.logical_not
        CodeObject.__init__(self, code, namespace, variables)

    def compile(self):
        super(NumpyCodeObject, self).compile()
        self.compiled_code = compile(self.code, '(string)', 'exec')

    def run(self):
        exec self.compiled_code in self.namespace
        # output variables should land in the variable name _return_values
        if '_return_values' in self.namespace:
            return self.namespace['_return_values']

runtime_targets['numpy'] = NumpyCodeObject
