#! /usr/bin/env python
'''
Brian2 setup script
'''

# isort:skip_file

import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from typing import List

# A Helper function to require cython extension
def require_cython_extension(module_path, module_name):
    """
    Create a cythonized Extension object from a .pyx source.
    """
    # File paths
    base_path = os.path.join(*module_path)
    pyx_file = os.path.join(base_path, f"{module_name}.pyx")

    if not os.path.exists(pyx_file):
        raise FileNotFoundError(f"Required Cython source not found: {pyx_file}")

    # Module name for setuptools
    full_module_name = ".".join(module_path + [module_name])

    ext = Extension(full_module_name, [pyx_file])

    return ext



class BuildExtWithNumpy(build_ext):
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
extensions : List[Extension]=[]

# Now Cython is required and no python fallback is possible
spike_queue_ext = require_cython_extension(
    module_path=["brian2", "synapses"],
    module_name="cythonspikequeue",
)
extensions.append(spike_queue_ext)


setup(ext_modules=extensions,
      cmdclass={'build_ext': BuildExtWithNumpy})
