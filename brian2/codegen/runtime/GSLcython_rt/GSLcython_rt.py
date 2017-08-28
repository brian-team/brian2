'''
Module containing the Cython CodeObject for code generation for integration using the ODE solver provided in the
GNU Scientific Library (GSL)
'''
import sys
from distutils.errors import CompileError

from brian2.core.preferences import prefs

from ..cython_rt import CythonCodeObject
from ...generators.GSL_generator import GSLCythonCodeGenerator
from ...generators.cython_generator import CythonCodeGenerator

__all__ = ['GSLCythonCodeObject', 'IntegrationError']

class GSLCompileError(Exception):
    pass

class IntegrationError(Exception):
    '''
    Error used to signify that GSL was unable to complete integration (only works for cython)
    '''
    pass

class GSLCythonCodeObject(CythonCodeObject):

    templater = CythonCodeObject.templater.derive('brian2.codegen.runtime.GSLcython_rt')

    # CodeGenerator that is used to do bulk of abstract_code --> language specific translation
    original_generator_class = CythonCodeGenerator
    generator_class = GSLCythonCodeGenerator

    def compile(self):
        self.libraries += ['gsl', 'gslcblas']
        self.headers += ['<stdio.h>', '<stdlib.h>', '<gsl/gsl_odeiv2.h>', '<gsl/gsl_errno.h>','<gsl/gsl_matrix.h>']
        if sys.platform == 'win32':
            self.define_macros += [('WIN32', '1'), ('GSL_DLL', '1')]
        if prefs.GSL.directory is not None:
            self.include_dirs += [prefs.GSL.directory]
        try:
            super(GSLCythonCodeObject, self).compile()
        except CompileError as err:
            raise GSLCompileError(("\nCompilation of files generated for integration with GSL has failed."
                                   "\nOne cause for this could be incorrect installation of GSL itself."
                                   "\nIf GSL is installed but Python cannot find the correct files, it is "
                                   "also possible to give the gsl directory manually by specifying "
                                   "prefs.GSL.directory = ..."))
