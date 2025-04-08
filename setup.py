#! /usr/bin/env python
'''
Brian2 setup script
'''

# isort:skip_file

import io
import sys
import os
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.errors import CompileError, PlatformError as DistutilsPlatformError

try:
    from Cython.Build import cythonize
    cython_available = True
except ImportError:
    cython_available = False


def has_option(name):
    try:
        sys.argv.remove('--%s' % name)
        return True
    except ValueError:
        pass
    # allow passing all cmd line options also as environment variables
    env_val = os.getenv(name.upper().replace('-', '_'), 'false').lower()
    if env_val == "true":
        return True
    return False


WITH_CYTHON = has_option('with-cython')
FAIL_ON_ERROR = has_option('fail-on-error')

pyx_fname = os.path.join('brian2', 'synapses', 'cythonspikequeue.pyx')
cpp_fname = os.path.join('brian2', 'synapses', 'cythonspikequeue.cpp')

if WITH_CYTHON or not os.path.exists(cpp_fname):
    fname = pyx_fname
    if not cython_available:
        if FAIL_ON_ERROR and WITH_CYTHON:
            raise RuntimeError('Compilation with Cython requested/necessary but '
                               'Cython is not available.')
        else:
            sys.stderr.write('Compilation with Cython requested/necessary but '
                             'Cython is not available.\n')
            fname = None
    if not os.path.exists(pyx_fname):
        if FAIL_ON_ERROR and WITH_CYTHON:
            raise RuntimeError(('Compilation with Cython requested/necessary but '
                                'Cython source file %s does not exist') % pyx_fname)
        else:
            sys.stderr.write(('Compilation with Cython requested/necessary but '
                              'Cython source file %s does not exist\n') % pyx_fname)
            fname = None
else:
    fname = cpp_fname

if fname is not None:
    extensions = [Extension("brian2.synapses.cythonspikequeue",
                            [fname],
                            include_dirs=[])]  # numpy include dir will be added later
    if fname == pyx_fname:
        extensions = cythonize(extensions)
else:
    extensions = []


class optional_build_ext(build_ext):
    '''
    This class allows the building of C extensions to fail and still continue
    with the building process. This ensures that installation never fails, even
    on systems without a C compiler, for example.
    If brian is installed in an environment where building C extensions
    *should* work, use the "--fail-on-error" option or set the environment
    variable FAIL_ON_ERROR to true.
    '''
    def build_extension(self, ext):
        import numpy
        numpy_incl = numpy.get_include()
        if hasattr(ext, 'include_dirs') and not numpy_incl in ext.include_dirs:
                ext.include_dirs.append(numpy_incl)
        try:
            build_ext.build_extension(self, ext)
        except (CompileError, DistutilsPlatformError) as ex:
            if FAIL_ON_ERROR:
                raise ex
            else:
                error_msg = ('Building %s failed (see error message(s) '
                             'above) -- pure Python version will be used '
                             'instead.') % ext.name
                sys.stderr.write('*' * len(error_msg) + '\n' +
                                 error_msg + '\n' +
                                 '*' * len(error_msg) + '\n')


setup(ext_modules=extensions,
      cmdclass={'build_ext': optional_build_ext})
