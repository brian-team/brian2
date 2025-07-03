#! /usr/bin/env python
'''
Brian2 setup script
'''

# isort:skip_file

import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext



try:
    from Cython.Build import cythonize
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

# A Helper function to require cython extension
def require_cython_extension(module_path, module_name, additional_sources=None, include_dirs=None, language='c'):
    """
    Create a Cython Extension object, preferring pre-generated .cpp files.

    Build strategy:
    1. If .cpp exists -> use it directly (fast, no Cython needed)
    2. If only .pyx exists -> cythonize it (requires Cython)
    3. If neither exists -> fail

    Args:
        module_path (list): Module path components, e.g. ["brian2", "synapses"]
        module_name (str): Extension name, e.g. "cythonspikequeue"
        additional_sources (list): Extra .cpp/.c files to include
        include_dirs (list): Additional include directories
        language (str): 'c' or 'c++'

    Returns:
        Extension: Ready-to-build extension object
    """
    additional_sources = additional_sources or []
    include_dirs = include_dirs or []

    # File paths
    base_path = os.path.join(*module_path)
    pyx_file = os.path.join(base_path, f"{module_name}.pyx")
    cpp_file = os.path.join(base_path, f"{module_name}.cpp")

    # Module name for setuptools
    full_module_name = ".".join(module_path + [module_name])

    # Determine source strategy
    if os.path.exists(cpp_file):
        sources = [cpp_file] + additional_sources
        needs_cythonize = False

    elif os.path.exists(pyx_file):
        if not CYTHON_AVAILABLE:
            raise RuntimeError(
                f"Cython source {pyx_file} found but Cython is not available. "
                "Install Cython or provide pre-generated .cpp file."
            )
        print(f"✓ Cythonizing source file: {pyx_file}")
        sources = [pyx_file] + additional_sources
        needs_cythonize = True

    else:
        raise FileNotFoundError(
            f"No source found for {full_module_name}. "
            f"Need either {pyx_file} or {cpp_file}"
        )

    # Validate additional sources exist
    for src in additional_sources:
        if not os.path.exists(src):
            raise FileNotFoundError(f"Additional source not found: {src}")


    ext = Extension(
        name=full_module_name,
        sources=sources,
        include_dirs=include_dirs,  # numpy include added later
        language=language
    )

    # Cythonize if needed
    if needs_cythonize:
        return cythonize([ext])[0]
    else:
        return ext


class OptionalBuildExt(build_ext):
    """
    Build Cython extensions with numpy include path.
    Fail if build fails.
    """
    def build_extension(self, ext):
        import numpy
        numpy_include = numpy.get_include()
        if hasattr(ext, 'include_dirs') and numpy_include not in ext.include_dirs:
            ext.include_dirs.append(numpy_include)
        build_ext.build_extension(self, ext)

# Collect Extensions
extensions =[]

# Now Cython is required and no python fallback is possible
spike_queue_ext = require_cython_extension(
    module_path=["brian2", "synapses"],
    module_name="cythonspikequeue",
    additional_sources=["brian2/synapses/cspikequeue.cpp"],  # Required C++ implementation
    include_dirs=["brian2/synapses"],  # For brian2 headers (stdint_compat.h, etc.)
    language='c++'
)
extensions.append(spike_queue_ext)


setup(ext_modules=extensions,
      cmdclass={'build_ext': OptionalBuildExt})
