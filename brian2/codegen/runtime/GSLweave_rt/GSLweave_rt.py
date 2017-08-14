from brian2.core.functions import DEFAULT_FUNCTIONS
from ..weave_rt import WeaveCodeObject
from ...generators.GSL_generator import GSLWeaveCodeGenerator
from ..weave_rt import WeaveCodeGenerator
from distutils.errors import CompileError

__all__ = ['GSLWeaveCodeObject']

class GSLWeaveCodeObject(WeaveCodeObject):

    templater = WeaveCodeObject.templater.derive('brian2.codegen.runtime.GSLweave_rt')

    original_generator_class = WeaveCodeGenerator
    generator_class = GSLWeaveCodeGenerator

    def compile(self):
        try:
            super(GSLWeaveCodeObject, self).compile()
            print('Got to the end of compile()')
        except Exception as err:
            raise
