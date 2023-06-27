#! /usr/bin/env python
'''
Brian2 setup script
'''

# isort:skip_file

import io
import sys
import os
import platform
from packaging.version import parse as parse_version
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.errors import CompileError, DistutilsPlatformError

REQUIRED_CYTHON_VERSION = '0.29'

try:
    import Cython
    if parse_version(Cython.__version__) < parse_version(REQUIRED_CYTHON_VERSION):
        raise ImportError('Cython version %s is too old' % Cython.__version__)
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
                               'Cython >= %s is not available.' % REQUIRED_CYTHON_VERSION)
        else:
            sys.stderr.write('Compilation with Cython requested/necessary but '
                             'Cython >= %s is not available.\n' % REQUIRED_CYTHON_VERSION)
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
    if (platform.system() == 'Linux' and
            platform.architecture()[0] == '32bit' and
            platform.machine() == 'x86_64'):
        # We are cross-compiling (most likely to build a 32Bit package for conda
        # on travis), set paths and flags for 32Bit explicitly
        print('Configuring compilation for cross-compilation to 32 Bit')
        extensions = [Extension("brian2.synapses.cythonspikequeue",
                                [fname],
                                include_dirs=[], # numpy include dir will be added later
                                library_dirs=['/lib32', '/usr/lib32'],
                                extra_compile_args=['-m32'],
                                extra_link_args=['-m32'])]
    else:
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


# Use readme file as long description
with io.open(os.path.join(os.path.dirname(__file__), 'README.rst'),
             encoding='utf-8') as f:
    long_description = f.read()

setup(ext_modules=extensions,
      cmdclass={'build_ext': optional_build_ext})
