'''
Cython automatic extension builder/manager

Inspired by IPython's Cython cell magics, see:
https://github.com/ipython/ipython/blob/master/IPython/extensions/cythonmagic.py
'''
import glob
import imp
import os
import shutil
import sys
import time
from threading import Thread

try:
    import msvcrt
except ImportError:
    msvcrt = None
    import fcntl

try:
    import hashlib
except ImportError:
    import md5 as hashlib

from distutils.core import Distribution, Extension
from distutils.command.build_ext import build_ext

import numpy
try:
    import Cython
    import Cython.Compiler as Cython_Compiler
    import Cython.Build as Cython_Build
    from Cython.Utils import get_cython_cache_dir as base_cython_cache_dir
except ImportError:
    Cython = None

from brian2.codegen.cpp_prefs import update_for_cross_compilation
from brian2.utils.logger import std_silent, get_logger
from brian2.utils.stringtools import deindent
from brian2.core.preferences import prefs

__all__ = ['cython_extension_manager']

logger = get_logger(__name__)


def get_cython_cache_dir():
    cache_dir = prefs.codegen.runtime.cython.cache_dir
    if cache_dir is None and Cython is not None:
        cache_dir = os.path.join(base_cython_cache_dir(), 'brian_extensions')
    return cache_dir


def get_cython_extensions():
    return {'.pyx', '.pxd', '.pyd', '.cpp', '.c', '.so', '.o', '.o.d', '.lock',
            '.dll', '.obj', '.exp', '.lib'}


def assure_lib_dir():
    lib_dir = get_cython_cache_dir()
    if '~' in lib_dir:
        lib_dir = os.path.expanduser(lib_dir)
    try:
        os.makedirs(lib_dir)
    except OSError:
        if not os.path.exists(lib_dir):
            raise IOError(
                "Couldn't create Cython cache directory '%s', try setting the "
                "cache directly with prefs.codegen.runtime.cython.cache_dir." % lib_dir)
    return lib_dir


def lock_file(fp, file_name):
    if msvcrt:
        msvcrt.locking(fp.fileno(), msvcrt.LK_RLCK,
                       os.stat(file_name).st_size)
    else:
        fcntl.flock(fp, fcntl.LOCK_EX)


def unlock_file(fp, file_name):
    if msvcrt:
        msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK,
                       os.stat(file_name).st_size)
    else:
        fcntl.flock(fp, fcntl.LOCK_UN)
    fp.close()


class CythonExtensionManager(object):
    def __init__(self):
        self._code_cache = {}

    def create_extension(self, code, force=False, name=None,
                         define_macros=None,
                         include_dirs=None,
                         library_dirs=None,
                         runtime_library_dirs=None,
                         extra_compile_args=None,
                         extra_link_args=None,
                         libraries=None,
                         compiler=None,
                         sources=None,
                         owner_name='',
                         ):
        """
        Compile Cython code into an extension that can be imported as a Python
        module.

        Returns
        -------
        module_name : str
            The name of the module (based on a hash of the code, Python version,
            etc.)
        build_process : `Process` or ``None``
            The process that compiles the module. Before loading the module,
            this process has to end, i.e. ``build_process.join()` has to be
            used. If no build process has been started because the module
            already exists, ``None`` will be returned
        """
        if sources is None:
            sources = []
        if define_macros is None:
            define_macros = []
        if include_dirs is None:
            include_dirs = []
        if library_dirs is None:
            library_dirs = []
        if extra_compile_args is None:
            extra_compile_args = []
        if extra_link_args is None:
            extra_link_args = []
        if libraries is None:
            libraries = []
        self._simplify_paths()

        if Cython is None:
            raise ImportError('Cython is not available')

        code = deindent(code)

        lib_dir = assure_lib_dir()

        numpy_version = '.'.join(numpy.__version__.split('.')[:2])  # Only use major.minor version
        key = code, sys.version_info, sys.executable, Cython.__version__, numpy_version
            
        if force:
            # Force a new module name by adding the current time to the
            # key which is hashed to determine the module name.
            key += time.time(),

        module_name = "_cython_magic_" + hashlib.md5(str(key).encode('utf-8')).hexdigest()
        if owner_name:
            logger.diagnostic('"{owner_name}" using Cython module "{module_name}"'.format(owner_name=owner_name,
                                                                                     module_name=module_name))
        # Nothing to do
        if module_name in self._code_cache:
            return module_name, None

        module_path = os.path.join(lib_dir, module_name + self.so_ext)

        lock_file_name = os.path.join(lib_dir, module_name + '.lock')
        lock_file_fp = open(lock_file_name, 'w')
        # Lock
        lock_file(lock_file_fp, lock_file_name)

        # We might just have acquired the lock after waiting for another
        # process to finish creating the module
        if module_name in self._code_cache:
            unlock_file(lock_file_fp, lock_file_name)
            return module_name, None

        # The module exists already, but has not yet been loaded into the
        # memory cache
        if os.path.isfile(module_path):
            unlock_file(lock_file_fp, lock_file_name)
            return module_name, None

        c_include_dirs = include_dirs
        if 'numpy' in code:
            c_include_dirs.append(numpy.get_include())

        # TODO: We should probably have a special folder just for header
        # files that are shared between different codegen targets
        import brian2.synapses as synapses
        synapses_dir = os.path.dirname(synapses.__file__)
        c_include_dirs.append(synapses_dir)

        pyx_file = os.path.join(lib_dir, module_name + '.pyx')
        with open(pyx_file, 'w') as f:
            f.write(code)

        update_for_cross_compilation(library_dirs,
                                     extra_compile_args,
                                     extra_link_args, logger=logger)
        for source in sources:
            if not source.lower().endswith('.pyx'):
                raise ValueError('Additional Cython source files need to '
                                 'have an .pyx ending')
            # Copy source and header file (if present) to library directory
            shutil.copyfile(source, os.path.join(lib_dir,
                                                 os.path.basename(source)))
            name_without_ext = os.path.splitext(os.path.basename(source))[0]
            header_name = name_without_ext + '.pxd'
            if os.path.exists(os.path.join(os.path.dirname(source), header_name)):
                shutil.copyfile(os.path.join(os.path.dirname(source), header_name),
                                os.path.join(lib_dir, header_name))
        final_sources = [os.path.join(lib_dir, os.path.basename(source))
                         for source in sources]
        p = Thread(target=self.build_module, name='build_{}'.format(module_name),
                   args=(c_include_dirs, compiler, define_macros,
                         extra_compile_args, extra_link_args,
                         final_sources, lib_dir, libraries, library_dirs,
                         module_name, pyx_file, runtime_library_dirs,
                         lock_file_name, lock_file_fp))
        p.start()
        # Note that the process will take care of unlocking the lock file!
        return module_name, p

    @property
    def so_ext(self):
        """The extension suffix for compiled modules."""
        try:
            return self._so_ext
        except AttributeError:
            self._so_ext = self._get_build_extension().get_ext_filename('')
            return self._so_ext

    def _clear_distutils_mkpath_cache(self):
        """clear distutils mkpath cache
        
        prevents distutils from skipping re-creation of dirs that have been removed
        """
        try:
            from distutils.dir_util import _path_created
        except ImportError:
            pass
        else:
            _path_created.clear()

    def _get_build_extension(self, compiler=None):
        self._clear_distutils_mkpath_cache()
        dist = Distribution()
        config_files = dist.find_config_files()
        try:
            config_files.remove('setup.cfg')
        except ValueError:
            pass
        dist.parse_config_files(config_files)
        build_extension = build_ext(dist)
        if compiler is not None:
            build_extension.compiler = compiler
        build_extension.finalize_options()
        return build_extension

    def build_module(self, c_include_dirs, compiler, define_macros,
                     extra_compile_args, extra_link_args, final_sources,
                     lib_dir, libraries, library_dirs, module_name, pyx_file,
                     runtime_library_dirs, lock_file_name, lock_file_fp):
        extension = Extension(
            name=module_name,
            sources=[pyx_file],
            define_macros=define_macros,
            include_dirs=c_include_dirs,
            library_dirs=library_dirs,
            runtime_library_dirs=runtime_library_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=libraries,
            language='c++')
        build_extension = self._get_build_extension(compiler=compiler)
        opts = dict(
            quiet=True,
            annotate=False,
            force=True,
        )
        # suppresses the output on stdout
        with std_silent():
            build_extension.extensions = Cython_Build.cythonize(
                [extension] + final_sources, **opts)
            build_extension.build_temp = os.path.dirname(pyx_file)
            build_extension.build_lib = lib_dir
            build_extension.run()
            if prefs['codegen.runtime.cython.delete_source_files']:
                # we can delete the source files to save disk space
                cpp_file = os.path.join(lib_dir, module_name + '.cpp')
                try:
                    os.remove(pyx_file)
                    os.remove(cpp_file)
                    temp_dir = os.path.join(lib_dir,
                                            os.path.dirname(pyx_file)[1:],
                                            module_name + '.*')
                    for fname in glob.glob(temp_dir):
                        os.remove(fname)
                except (OSError, IOError) as ex:
                    logger.debug('Deleting Cython source files failed with '
                                 'error: %s' % str(ex))
        # unlock the file lock
        unlock_file(lock_file_fp, lock_file_name)

    def get_module(self, module_name):
        if module_name in self._code_cache:
            return self._code_cache[module_name]
        lib_dir = assure_lib_dir()
        module_path = os.path.join(lib_dir, module_name + self.so_ext)
        # Temporarily insert the Cython directory to the Python path so that
        # code importing from an external module that was declared via
        # sources works
        sys.path.insert(0, lib_dir)
        module = imp.load_dynamic(module_name, module_path)
        sys.path.pop(0)
        self._code_cache[module_name] = module
        return module

    def _simplify_paths(self):
        if 'lib' in os.environ:
            os.environ['lib'] = simplify_path_env_var(os.environ['lib'])
        if 'include' in os.environ:
            os.environ['include'] = simplify_path_env_var(os.environ['include'])


cython_extension_manager = CythonExtensionManager()


def simplify_path_env_var(path):
    allpaths = path.split(os.pathsep)
    knownpaths = set()
    uniquepaths = []
    for p in allpaths:
        if p not in knownpaths:
            knownpaths.add(p)
            uniquepaths.append(p)
    return os.pathsep.join(uniquepaths)

