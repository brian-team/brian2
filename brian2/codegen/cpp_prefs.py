'''
Preferences related to C++ compilation

Preferences
--------------------
.. document_brian_prefs:: codegen.cpp

'''
from distutils.ccompiler import get_default_compiler

from cpuinfo import cpuinfo

from brian2.core.preferences import prefs, BrianPreference

__all__ = ['get_compiler_and_args']

# Try to get architecture information to get the best compiler setting for
# Windows
msvc_arch_flag = ''
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
        default=['-w', '-O3', '-ffast-math', '-march=native'],
        docs='''
        Extra compile arguments to pass to GCC compiler
        '''
        ),
    extra_compile_args_msvc=BrianPreference(
        default=['/Ox', '/EHsc', '/w', '/fp:fast', msvc_arch_flag],
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
