'''
Preferences related to C++ compilation

Preferences
--------------------
.. document_brian_prefs:: codegen.cpp

'''
from distutils.ccompiler import get_default_compiler

from brian2.core.preferences import prefs, BrianPreference
from .codeobject import sys_info

__all__ = ['get_compiler_and_args']

# Try to get architecture information to get the best compiler setting for
# Windows
msvc_arch_flag = ''
try:
    from cpuinfo import cpuinfo
    res = cpuinfo.get_cpu_info()
    # Note that this overwrites the arch_flag, i.e. only the best option will
    # be used
    if 'sse' in res['flags']:
        msvc_arch_flag = '/arch:SSE'
    if 'sse2' in res['flags']:
        msvc_arch_flag = '/arch:SSE2'
    if 'avx' in res['flags']:
        msvc_arch_flag = '/arch:AVX'
    if 'avx2' in res['flags']:
        msvc_arch_flag = '/arch:AVX2'
except Exception:
    pass  # apparently it does not always work on appveyor


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
