"""
Preferences related to C++ compilation

Preferences
--------------------
.. document_brian_prefs:: codegen.cpp

"""

import distutils
import json
import os
import platform
import re
import socket
import struct
import subprocess
import sys
import tempfile
from distutils.ccompiler import get_default_compiler

from brian2.core.preferences import BrianPreference, prefs
from brian2.utils.filetools import ensure_directory
from brian2.utils.logger import get_logger, std_silent

__all__ = ["get_compiler_and_args", "get_msvc_env", "compiler_supports_c99", "C99Check"]


logger = get_logger(__name__)

# default_buildopts stores default build options for Gcc compiler
default_buildopts = []

# Try to get architecture information to get the best compiler setting for
# Windows
msvc_arch_flag = ""
if platform.system() == "Windows":
    flags = None
    previously_stored_flags = None

    # Check whether we've already stored the CPU flags previously
    user_dir = os.path.join(os.path.expanduser("~"), ".brian")
    ensure_directory(user_dir)
    flag_file = os.path.join(user_dir, "cpu_flags.txt")
    hostname = socket.gethostname()
    if os.path.isfile(flag_file):
        try:
            with open(flag_file, encoding="utf-8") as f:
                previously_stored_flags = json.load(f)
            if hostname not in previously_stored_flags:
                logger.debug("Ignoring stored CPU flags for a different host")
            else:
                flags = previously_stored_flags[hostname]
        except OSError as ex:
            logger.debug(
                f'Opening file "{flag_file}" to get CPU flags failed with error'
                f' "{str(ex)}".'
            )

    if flags is None:  # If we don't have stored info, run get_cpu_flags.py
        get_cpu_flags_script = os.path.join(
            os.path.dirname(__file__), "get_cpu_flags.py"
        )
        get_cpu_flags_script = os.path.abspath(get_cpu_flags_script)
        try:
            output = subprocess.check_output(
                [sys.executable, get_cpu_flags_script],
                text=True,
                encoding="utf-8",
            )
            flags = json.loads(output)
            # Store flags to a file so we don't have to call cpuinfo next time
            try:
                if previously_stored_flags is not None:
                    to_store = previously_stored_flags
                    to_store[hostname] = flags
                else:
                    to_store = {hostname: flags}
                with open(flag_file, "w", encoding="utf-8") as f:
                    json.dump(to_store, f)
            except OSError as ex:
                logger.debug(
                    f'Writing file "{flag_file}" to store CPU flags failed with error'
                    f' "{str(ex)}".'
                )
        except subprocess.CalledProcessError as ex:
            logger.debug(
                "Could not determine optimized MSVC flags, get_cpu_flags failed with:"
                f" {str(ex)}"
            )

    if flags is not None:
        # Note that this overwrites the arch_flag, i.e. only the best option
        # will be used
        if "sse" in flags:
            msvc_arch_flag = "/arch:SSE"
        if "sse2" in flags:
            msvc_arch_flag = "/arch:SSE2"
        if "avx" in flags:
            msvc_arch_flag = "/arch:AVX"
        if "avx2" in flags:
            msvc_arch_flag = "/arch:AVX2"

else:
    # Optimized default build options for a range a CPU architectures
    machine = os.uname().machine
    if re.match("^(x86_64|aarch64|arm.*|s390.*|i.86.*)$", machine):
        default_buildopts = [
            "-w",
            "-O3",
            "-ffast-math",
            "-fno-finite-math-only",
            "-march=native",
            "-std=c++11",
        ]
    elif re.match("^(alpha|ppc.*|sparc.*)$", machine):
        default_buildopts = [
            "-w",
            "-O3",
            "-ffast-math",
            "-fno-finite-math-only",
            "-mcpu=native",
            "-mtune=native",
            "-std=c++11",
        ]
    elif re.match("^(parisc.*|riscv.*|mips.*|loong64.*)$", machine):
        default_buildopts = [
            "-w",
            "-O3",
            "-ffast-math",
            "-fno-finite-math-only",
            "-std=c++11",
        ]
    else:
        default_buildopts = ["-w"]

if os.environ.get("READTHEDOCS", "False").lower() == "true":
    # We are getting imported during a documentation build. Set a fake prefix
    # to avoid having the name of the local environment in the documentation
    sys_prefix = "/path/to/your/Python/environment"
else:
    sys_prefix = sys.prefix

if sys.platform == "win32":
    prefix_dir = os.path.join(sys_prefix, "Library")
else:
    prefix_dir = sys_prefix

# Preferences
prefs.register_preferences(
    "codegen.cpp",
    "C++ compilation preferences",
    compiler=BrianPreference(
        default="",
        docs="""
        Compiler to use (uses default if empty).
        Should be ``'unix'`` or ``'msvc'``.

        To specify a specific compiler binary on unix systems, set the `CXX` environment
        variable instead.
        """,
    ),
    extra_compile_args=BrianPreference(
        default=None,
        validator=lambda v: True,
        docs="""
        Extra arguments to pass to compiler (if None, use either
        ``extra_compile_args_gcc`` or ``extra_compile_args_msvc``).
        """,
    ),
    extra_compile_args_gcc=BrianPreference(
        default=default_buildopts,
        docs="""
        Extra compile arguments to pass to GCC compiler
        """,
    ),
    extra_compile_args_msvc=BrianPreference(
        default=["/Ox", "/w", msvc_arch_flag, "/MP"],
        docs="""
        Extra compile arguments to pass to MSVC compiler (the default
        ``/arch:`` flag is determined based on the processor architecture)
        """,
    ),
    extra_link_args=BrianPreference(
        default=[],
        docs="""
        Any extra platform- and compiler-specific information to use when
        linking object files together.
        """,
    ),
    include_dirs=BrianPreference(
        default=[os.path.join(prefix_dir, "include")],
        docs="""
        Include directories to use.
        The default value is ``$prefix/include`` (or ``$prefix/Library/include``
        on Windows), where ``$prefix`` is Python's site-specific directory
        prefix as returned by `sys.prefix`. This will make compilation use
        library files installed into a conda environment.
        """,
    ),
    library_dirs=BrianPreference(
        default=[os.path.join(prefix_dir, "lib")],
        docs="""
        List of directories to search for C/C++ libraries at link time.
        The default value is ``$prefix/lib`` (or ``$prefix/Library/lib``
        on Windows), where ``$prefix`` is Python's site-specific directory
        prefix as returned by `sys.prefix`. This will make compilation use
        library files installed into a conda environment.
        """,
    ),
    runtime_library_dirs=BrianPreference(
        default=[os.path.join(prefix_dir, "lib")] if sys.platform != "win32" else [],
        docs="""
        List of directories to search for C/C++ libraries at run time.
        The default value is ``$prefix/lib`` (not used on Windows), where
        ``$prefix`` is Python's site-specific directory prefix as returned by
        `sys.prefix`. This will make compilation use library files installed
        into a conda environment.
        """,
    ),
    libraries=BrianPreference(
        default=[],
        docs="""
        List of library names (not filenames or paths) to link against.
        """,
    ),
    headers=BrianPreference(
        default=[],
        docs="""
        A list of strings specifying header files to use when compiling the
        code. The list might look like ["<vector>","'my_header'"]. Note that
        the header strings need to be in a form than can be pasted at the end
        of a #include statement in the C++ code.
        """,
    ),
    define_macros=BrianPreference(
        default=[],
        docs="""
        List of macros to define; each macro is defined using a 2-tuple,
        where 'value' is either the string to define it to or None to
        define it without a particular value (equivalent of "#define
        FOO" in source or -DFOO on Unix C compiler command line).
        """,
    ),
    msvc_vars_location=BrianPreference(
        default="",
        docs="""
        Location of the MSVC command line tool (or search for best by default).
        """,
    ),
    msvc_architecture=BrianPreference(
        default="",
        docs="""
        MSVC architecture name (or use system architectue by default).

        Could take values such as x86, amd64, etc.
        """,
    ),
)


# check whether compiler supports a flag
# Adapted from https://github.com/pybind/pybind11/
def _determine_flag_compatibility(compiler, flagname):
    import tempfile
    from distutils.errors import CompileError

    with (
        tempfile.TemporaryDirectory(prefix="brian_flag_test_") as temp_dir,
        std_silent(),
    ):
        fname = os.path.join(temp_dir, "flag_test.cpp")
        with open(fname, "w") as f:
            f.write("int main (int argc, char **argv) { return 0; }")
        try:
            compiler.compile([fname], output_dir=temp_dir, extra_postargs=[flagname])
        except CompileError:
            logger.warn(f"Removing unsupported flag '{flagname}' from compiler flags.")
            return False
    return True


_compiler_flag_compatibility = {}


def has_flag(compiler, flagname):
    if compiler.compiler_type == "msvc":
        # MSVC does not raise an error for illegal flags, so determining
        # whether it accepts a flag would mean parsing the output for warnings
        # This is non-trivial so we don't do it (the main reason to check
        # flags in the first place are differences between gcc and clang)
        return True
    else:
        compiler_exe = " ".join(compiler.executables["compiler_cxx"])

    if (compiler_exe, flagname) not in _compiler_flag_compatibility:
        compatibility = _determine_flag_compatibility(compiler, flagname)
        _compiler_flag_compatibility[(compiler_exe, flagname)] = compatibility

    return _compiler_flag_compatibility[(compiler_exe, flagname)]


def get_compiler_and_args():
    """
    Returns the computed compiler and compilation flags
    """
    compiler = prefs["codegen.cpp.compiler"]
    if compiler == "":
        compiler = get_default_compiler()
    extra_compile_args = prefs["codegen.cpp.extra_compile_args"]
    if extra_compile_args is None:
        if compiler in ("gcc", "unix"):
            extra_compile_args = prefs["codegen.cpp.extra_compile_args_gcc"]
        elif compiler == "msvc":
            extra_compile_args = prefs["codegen.cpp.extra_compile_args_msvc"]
        else:
            extra_compile_args = []
            logger.warn(f"Unsupported compiler '{compiler}'.")

    from distutils.ccompiler import new_compiler
    from distutils.sysconfig import customize_compiler

    compiler_obj = new_compiler(compiler=compiler, verbose=0)
    customize_compiler(compiler_obj)
    extra_compile_args = [
        flag for flag in extra_compile_args if has_flag(compiler_obj, flag)
    ]

    return compiler, extra_compile_args


_msvc_env = None


def get_msvc_env():
    try:
        from setuptools.msvc import msvc14_get_vc_env as _get_vc_env
    except ImportError:  # Setuptools 0.74.0 removed this function
        try:
            from distutils.compilers.C.msvc import _get_vc_env
        except ImportError:  # Things keep moving around in distutils/setuptools
            from distutils._msvccompiler import _get_vc_env

    global _msvc_env
    arch_name = prefs["codegen.cpp.msvc_architecture"]
    if arch_name == "":
        bits = struct.calcsize("P") * 8
        if bits == 64:
            arch_name = "x86_amd64"
        else:
            arch_name = "x86"
    # Manual specification of vcvarsall.bat location by the user
    vcvars_loc = prefs["codegen.cpp.msvc_vars_location"]
    if vcvars_loc:
        vcvars_cmd = f'"{vcvars_loc}" {arch_name}'
        return None, vcvars_cmd

    # Search for MSVC environment if not already cached
    if _msvc_env is None:
        try:
            _msvc_env = _get_vc_env(arch_name)
        except distutils.errors.DistutilsPlatformError:
            raise OSError(
                "Cannot find Microsoft Visual Studio, You "
                "can try to set the path to vcvarsall.bat "
                "via the codegen.cpp.msvc_vars_location "
                "preference explicitly."
            )
    return _msvc_env, None


_compiler_supports_c99 = None


def compiler_supports_c99():
    global _compiler_supports_c99
    if _compiler_supports_c99 is None:
        if platform.system() == "Windows":
            fd, tmp_file = tempfile.mkstemp(suffix=".cpp")
            os.write(
                fd,
                b"""
            #if _MSC_VER < 1800
            #error
            #endif
            """,
            )
            os.close(fd)
            msvc_env, vcvars_cmd = get_msvc_env()
            if vcvars_cmd:
                cmd = f"{vcvars_cmd} && cl /E {tmp_file} > NUL 2>&1"
            else:
                os.environ.update(msvc_env)
                cmd = f"cl /E {tmp_file} > NUL 2>&1"
            return_value = os.system(cmd)
            _compiler_supports_c99 = return_value == 0
            os.remove(tmp_file)
        else:
            CC = os.environ.get("CC", "cc")
            cmd = (
                'echo "#if (__STDC_VERSION__ < 199901L)\n#error\n#endif" | '
                f"'{CC}' -xc -E - > /dev/null 2>&1"
            )
            return_value = os.system(cmd)
            _compiler_supports_c99 = return_value == 0
    return _compiler_supports_c99


class C99Check:
    """
    Helper class to create objects that can be passed as an ``availability_check`` to
    a `FunctionImplementation`.
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        if not compiler_supports_c99():
            raise NotImplementedError(
                f'The "{self.name}" function needs C99 compiler support'
            )
