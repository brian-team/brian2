'''
Brian 2
'''
# Import setuptools to do some monkey patching of distutils, necessary for
# working weave/Cython on Windows with the Python for C++ compiler
import setuptools as _setuptools

def _check_dependencies():
    '''Check basic dependencies'''
    import sys
    missing = []
    try:
        import numpy
    except ImportError as ex:
        sys.stderr.write('Importing numpy failed: %s\n' % ex)
        missing.append('numpy')
    try:
        import sympy
    except ImportError as ex:
        sys.stderr.write('Importing sympy failed: %s\n' % ex)
        missing.append('sympy')
    try:
        import pyparsing
    except ImportError as ex:
        sys.stderr.write('Importing pyparsing failed: %s\n' % ex)
        missing.append('pyparsing')
    try:
        import jinja2
    except ImportError as ex:
        sys.stderr.write('Importing Jinja2 failed: %s\n' % ex)
        missing.append('jinja2')

    if len(missing):
        raise ImportError('Some required dependencies are missing:\n' + ', '.join(missing))

_check_dependencies()

try:
    from pylab import *
except ImportError:
    # Do the non-matplotlib pylab imports manually
    from numpy import *
    from numpy.fft import *
    from numpy.random import *
    from numpy.linalg import *
    import numpy.ma as ma
    # don't let numpy's datetime hide stdlib
    import datetime

# Make sure that Brian's unit-aware functions are used, even when directly
# using names prefixed with numpy or np
import brian2.numpy_ as numpy
import brian2.numpy_ as np

# delete some annoying names from the namespace
if 'x' in globals():
    del x
if 'f' in globals():
    del f
if 'rate' in globals():
    del rate

__docformat__ = "restructuredtext en"

__version__ = '2.2.1'
__release_date__ = '2018-11-19'

from brian2.only import *

# Check for outdated dependency versions
def _check_dependency_version(name, version):
    from distutils.version import LooseVersion
    from core.preferences import prefs
    from utils.logger import get_logger
    import sys
    logger = get_logger(__name__)

    module = sys.modules[name]
    if not isinstance(module.__version__, basestring):  # mocked module
        return
    if not LooseVersion(module.__version__) >= LooseVersion(version):
        message = '%s is outdated (got version %s, need version %s)' % (name,
                                                                        module.__version__,
                                                                        version)
        if prefs.core.outdated_dependency_error:
            raise ImportError(message)
        else:

            logger.warn(message, 'outdated_dependency')


def _check_dependency_versions():
    for name, version in [('numpy',  '1.10'),
                          ('sympy',  '0.7.6'),
                          ('jinja2', '2.7')]:
        _check_dependency_version(name, version)

_check_dependency_versions()

# Initialize the logging system
BrianLogger.initialize()
logger = get_logger(__name__)

# Check the caches
def _get_size_recursively(dirname):
    import os
    total_size = 0
    for dirpath, _, filenames in os.walk(dirname):
        for fname in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, fname))
    return total_size

#: Stores the cache directory for code generation targets
_cache_dirs_and_extensions = {}

def check_cache(target):
    cache_dir, _ = _cache_dirs_and_extensions.get(target, (None, None))
    if cache_dir is None:
        return
    size = _get_size_recursively(cache_dir)
    size_in_mb = int(round(size/1024./1024.))
    if size_in_mb > prefs.codegen.max_cache_dir_size:
        logger.info('Cache size for target "{target}": {size} MB.\n'
                    'You can call "clear_cache(\'{target}\')" to delete all '
                    'files from the cache or manually delete files in the '
                    '"{cache_dir}" directory.'.format(target=target,
                                                      size=size_in_mb,
                                                      cache_dir=cache_dir))
    else:
        logger.debug('Cache size for target "%s": %s MB' % (target, size_in_mb))


def clear_cache(target):
    '''
    Clears the on-disk cache with the compiled files for a given code generation
    target.

    Parameters
    ----------
    target : str
        The code generation target (e.g. ``'weave'`` or ``'cython'``)

    Raises
    ------
    ValueError
        If the given code generation target does not have an on-disk cache
    IOError
        If the cache directory contains unexpected files, suggesting that
        deleting it would also delete files unrelated to the cache.
    '''
    import os
    import shutil
    cache_dir, extensions = _cache_dirs_and_extensions.get(target, (None, None))
    if cache_dir is None:
        raise ValueError('No cache directory registered for target '
                         '"%s".' % target)
    cache_dir = os.path.abspath(cache_dir)  # just to make sure...
    for folder, _, files in os.walk(cache_dir):
        for f in files:
            for ext in extensions:
                if f.endswith(ext):
                    break
            else:
                raise IOError("The cache directory for target '{}' contains "
                              "the file '{}' of an unexpected type and "
                              "will therefore not be removed. Delete files in "
                              "'{}' manually".format(target,
                                                     os.path.join(folder, f),
                                                     cache_dir))

    logger.debug('Clearing cache for target "%s" (directory "%s").' %
                 (target, cache_dir))
    shutil.rmtree(cache_dir)


def _check_caches():
    from brian2.codegen.runtime.weave_rt.weave_rt import get_weave_cache_dir
    from brian2.codegen.runtime.cython_rt.extension_manager import get_cython_cache_dir
    from brian2.codegen.runtime.weave_rt.weave_rt import get_weave_extensions
    from brian2.codegen.runtime.cython_rt.extension_manager import get_cython_extensions

    for target, (dirname, extensions) in [('weave', (get_weave_cache_dir(), get_weave_extensions())),
                                         ('cython', (get_cython_cache_dir(), get_cython_extensions()))]:
        _cache_dirs_and_extensions[target] = (dirname, extensions)
        if prefs.codegen.max_cache_dir_size > 0:
            check_cache(target)

_check_caches()
