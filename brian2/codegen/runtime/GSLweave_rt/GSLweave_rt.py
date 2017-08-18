from brian2.core.functions import DEFAULT_FUNCTIONS
from ..weave_rt import WeaveCodeObject
from ...generators.GSL_generator import GSLWeaveCodeGenerator
from ..weave_rt import WeaveCodeGenerator
from weave.build_tools import CompileError

__all__ = ['GSLWeaveCodeObject']

class GSLCompileError(Exception):
    pass

class GSLWeaveCodeObject(WeaveCodeObject):

    templater = WeaveCodeObject.templater.derive('brian2.codegen.runtime.GSLweave_rt')

    original_generator_class = WeaveCodeGenerator
    generator_class = GSLWeaveCodeGenerator

    def run(self):
        try:
            super(GSLWeaveCodeObject, self).run()
        except CompileError as err:
            raise GSLCompileError(("\nCompilation of files generated for integration with GSL has failed."
                                   "\nOne cause for this could be incorrect installation of GSL itself."
                                   "\nIf GSL is installed but Python cannot find the correct files, it is "
                                   "also possible to give the gsl directory manually by specifying "
                                   "prefs.GSL.directory = ..."))
