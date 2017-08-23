from brian2.core.preferences import prefs

from ..weave_rt import WeaveCodeObject
from ...generators.GSL_generator import GSLWeaveCodeGenerator
from ..weave_rt import WeaveCodeGenerator
try:
    from scipy.weave.build_tools import CompileError
except ImportError:
    from weave.build_tools import CompileError

__all__ = ['GSLWeaveCodeObject']

class GSLCompileError(Exception):
    pass

class GSLWeaveCodeObject(WeaveCodeObject):

    templater = WeaveCodeObject.templater.derive('brian2.codegen.runtime.GSLweave_rt')

    original_generator_class = WeaveCodeGenerator
    generator_class = GSLWeaveCodeGenerator

    def run(self):
        self.libraries += ['gsl', 'gslcblas']
        self.headers += ['<stdio.h>', '<stdlib.h>', '<gsl/gsl_odeiv2.h>', '<gsl/gsl_errno.h>','<gsl/gsl_matrix.h>']
        if prefs.GSL.directory is not None:
            self.include_dirs += [prefs.GSL.directory]
        try:
            super(GSLWeaveCodeObject, self).run()
        except CompileError as err:
            raise GSLCompileError(("\nCompilation of files generated for integration with GSL has failed."
                                   "\nOne cause for this could be incorrect installation of GSL itself."
                                   "\nIf GSL is installed but Python cannot find the correct files, it is "
                                   "also possible to give the gsl directory manually by specifying "
                                   "prefs.GSL.directory = ..."))
