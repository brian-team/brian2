#! /usr/bin/env python
'''
Brian2 setup script
'''

# isort:skip_file

import os
import numpy
from setuptools import setup, Extension
from typing import List

# A Helper function to require cython extension
def require_cython_extension(module_path, module_name,extra_include_dirs=None):
    """
    Create a cythonized Extension object from a .pyx source.
    """
    # File paths
    base_path = os.path.join(*module_path)
    pyx_file = os.path.join(base_path, f"{module_name}.pyx")

    # Module name for setuptools
    full_module_name = ".".join(module_path + [module_name])

    include_dirs = [numpy.get_include()]
    if extra_include_dirs:
        include_dirs.extend(extra_include_dirs)

    ext = Extension(full_module_name, [pyx_file],include_dirs=include_dirs)
    return ext


# Collect Extensions
extensions : List[Extension]=[]

# Now Cython is required and no python fallback is possible
spike_queue_ext = require_cython_extension(
    module_path=["brian2", "synapses"],
    module_name="cythonspikequeue",
)

extensions.append(spike_queue_ext)

dyanamic_array_ext = require_cython_extension(
    module_path=["brian2", "memory"],
    module_name="cythondynamicarray",
    extra_include_dirs=["brian2/devices/cpp_standalone/brianlib"]
)

extensions.append(dyanamic_array_ext)


setup(ext_modules=extensions)
