'''
Preferences related to C++ compilation

Preferences
--------------------
.. document_brian_prefs:: codegen.cpp

'''
from distutils.ccompiler import get_default_compiler
import json
import os
import platform
import socket
import subprocess
import sys

from brian2.core.preferences import prefs, BrianPreference
from brian2.utils.logger import get_logger

from .codeobject import sys_info

__all__ = ['get_compiler_and_args']


logger = get_logger(__name__)

# Try to get architecture information to get the best compiler setting for
# Windows
msvc_arch_flag = ''
if platform.system() == 'Windows':
    flags = None
    previously_stored_flags = None

    # Check whether we've already stored the CPU flags previously
    user_dir = os.path.join(os.path.expanduser('~'), '.brian')
    flag_file = os.path.join(user_dir, 'cpu_flags.txt')
    hostname = socket.gethostname()
    if os.path.isfile(flag_file):
        try:
            with open(flag_file, 'r') as f:
                previously_stored_flags = json.load(f, encoding='utf-8')
            if hostname not in previously_stored_flags:
                logger.debug('Ignoring stored CPU flags for a different host')
            else:
                flags = previously_stored_flags[hostname]
        except (IOError, OSError) as ex:
            logger.debug('Opening file "{}" to get CPU flags failed with error '
                         '"{}".'.format(flag_file, str(ex)))

    if flags is None:  # If we don't have stored info, run get_cpu_flags.py
        get_cpu_flags_script = os.path.join(os.path.dirname(__file__),
                                            'get_cpu_flags.py')
        get_cpu_flags_script = os.path.abspath(get_cpu_flags_script)
        try:
            output = subprocess.check_output([sys.executable,
                                              get_cpu_flags_script],
                                             universal_newlines=True)
            flags = json.loads(output)
            # Store flags to a file so we don't have to call cpuinfo next time
            try:
                if previously_stored_flags is not None:
                    to_store = previously_stored_flags
                    to_store[hostname] = flags
                else:
                    to_store = {hostname: flags}
                with open(flag_file, 'w') as f:
                    json.dump(to_store, f, encoding='utf-8')
            except (IOError, OSError) as ex:
                logger.debug('Writing file "{}" to store CPU flags failed with '
                             'error "{}".'.format(flag_file, str(ex)))
        except subprocess.CalledProcessError as ex:
            logger.debug('Could not determine optimized MSVC flags, '
                         'get_cpu_flags failed with: %s' % (str(ex)))

    if flags is not None:
        # Note that this overwrites the arch_flag, i.e. only the best option
        # will be used
        if 'sse' in flags:
            msvc_arch_flag = '/arch:SSE'
        if 'sse2' in flags:
            msvc_arch_flag = '/arch:SSE2'
        if 'avx' in flags:
            msvc_arch_flag = '/arch:AVX'
        if 'avx2' in flags:
            msvc_arch_flag = '/arch:AVX2'

# Preferences
prefs.register_preferences(
    'codegen.cpp',
    'C++ compilation preferences',
    compiler = BrianPreference(
        default='',
        docs='''
        Compiler to use (uses default if empty)
        
        Should be gcc or msvc.
        '''
        ),
    extra_compile_args=BrianPreference(
        default=None,
        validator=lambda v: True,
        docs='''
        Extra arguments to pass to compiler (if None, use either
        ``extra_compile_args_gcc`` or ``extra_compile_args_msvc``).
        '''
        ),
    extra_compile_args_gcc=BrianPreference(
        default=['-w', '-O3', '-ffast-math', '-fno-finite-math-only', '-march=native'],
        docs='''
        Extra compile arguments to pass to GCC compiler
        '''
        ),
    extra_compile_args_msvc=BrianPreference(
        default=['/Ox', '/w', msvc_arch_flag, '/MP'],
        docs='''
        Extra compile arguments to pass to MSVC compiler (the default
        ``/arch:`` flag is determined based on the processor architecture)
        '''
        ),
    extra_link_args=BrianPreference(
        default=[],
        docs='''
        Any extra platform- and compiler-specific information to use when
        linking object files together.
        '''
    ),
    include_dirs=BrianPreference(
        default=[],
        docs='''
        Include directories to use. Note that ``$prefix/include`` will be
        appended to the end automatically, where ``$prefix`` is Python's
        site-specific directory prefix as returned by `sys.prefix`.
        '''
        ),
    library_dirs=BrianPreference(
        default=[],
        docs='''
        List of directories to search for C/C++ libraries at link time.
        Note that ``$prefix/lib`` will be appended to the end automatically,
        where ``$prefix`` is Python's site-specific directory prefix as returned
        by `sys.prefix`.
        '''
    ),
    runtime_library_dirs=BrianPreference(
        default=[],
        docs='''
        List of directories to search for C/C++ libraries at run time.
        '''
    ),
    libraries=BrianPreference(
        default=[],
        docs='''
        List of library names (not filenames or paths) to link against.
        '''
    ),
    headers=BrianPreference(
        default=[],
        docs='''
        A list of strings specifying header files to use when compiling the
        code. The list might look like ["<vector>","'my_header'"]. Note that
        the header strings need to be in a form than can be pasted at the end
        of a #include statement in the C++ code.
        '''
    ),
    define_macros=BrianPreference(
        default=[],
        docs='''
        List of macros to define; each macro is defined using a 2-tuple,
        where 'value' is either the string to define it to or None to
        define it without a particular value (equivalent of "#define
        FOO" in source or -DFOO on Unix C compiler command line).
        '''
    ),
    msvc_vars_location=BrianPreference(
        default='',
        docs='''
        Location of the MSVC command line tool (or search for best by default).
        '''),
    msvc_architecture=BrianPreference(
        default='',
        docs='''
        MSVC architecture name (or use system architectue by default).
        
        Could take values such as x86, amd64, etc.
        '''),
    )


def get_compiler_and_args():
    '''
    Returns the computed compiler and compilation flags
    '''
    compiler = prefs['codegen.cpp.compiler']
    if compiler == '':
        compiler = get_default_compiler()
    extra_compile_args = prefs['codegen.cpp.extra_compile_args']
    if extra_compile_args is None:
        if compiler in ('gcc', 'unix'):
            extra_compile_args = prefs['codegen.cpp.extra_compile_args_gcc']
        if compiler == 'msvc':
            extra_compile_args = prefs['codegen.cpp.extra_compile_args_msvc']
    return compiler, extra_compile_args


def update_for_cross_compilation(library_dirs, extra_compile_args,
                                 extra_link_args, logger=None):
    '''
    Update the compiler arguments to allow cross-compilation for 32bit on a
    64bit Linux system. Uses the provided ``logger`` to print an INFO message
    and modifies the provided lists in-place.

    Parameters
    ----------
    library_dirs : list
        List of library directories (will be modified in-place).
    extra_compile_args : list
        List of extra compile args (will be modified in-place).
    extra_link_args : list
        List of extra link args (will be modified in-place).
    logger : `BrianLogger`, optional
        The logger to use for the INFO message. Defaults to ``None`` (no
        message).
    '''
    if (sys_info['system'] == 'Linux' and
                sys_info['architecture'][0] == '32bit' and
                sys_info['machine'] == 'x86_64'):
        # We are cross-compiling to 32bit on a 64bit platform
        if logger is not None:
            logger.info('Cross-compiling to 32bit on a 64bit platform, a set '
                        'of standard compiler options will be appended for '
                        'this purpose (note that you need to have a 32bit '
                        'version of the standard library for this to work).',
                        '64bit_to_32bit',
                        once=True)
        library_dirs += ['/lib32', '/usr/lib32']
        extra_compile_args += ['-m32']
        extra_link_args += ['-m32']
