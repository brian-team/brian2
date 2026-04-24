"""
Brian 2
"""

import logging
import signal
import re

def _check_dependencies():
    """Check basic dependencies"""
    import sys

    missing = []
    try:
        import numpy
    except ImportError as ex:
        sys.stderr.write(f"Importing numpy failed: '{ex}'\n")
        missing.append("numpy")
    try:
        import sympy
    except ImportError as ex:
        sys.stderr.write(f"Importing sympy failed: '{ex}'\n")
        missing.append("sympy")
    try:
        import pyparsing
    except ImportError as ex:
        sys.stderr.write(f"Importing pyparsing failed: '{ex}'\n")
        missing.append("pyparsing")
    try:
        import jinja2
    except ImportError as ex:
        sys.stderr.write(f"Importing Jinja2 failed: '{ex}'\n")
        missing.append("jinja2")

    if len(missing):
        raise ImportError(
            f"Some required dependencies are missing:\n{', '.join(missing)}"
        )

_check_dependencies()

# FIXED: Removed wildcard import from inside function to avoid SyntaxError
# Instead of 'from pylab import *', we import numpy essentials directly 
# and keep pylab optional to avoid slow Matplotlib load at startup.
try:
    import numpy as numpy
    import brian2.numpy_ as np
except ImportError:
    pass

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    try:
        from setuptools_scm import get_version

        __version__ = get_version(
            root="..",
            relative_to=__file__,
            version_scheme="post-release",
            local_scheme="no-local-version",
        )
        # FIXED: Use re.findall to safely extract version numbers
        # setuptools_scm can generate versions like '2.10.post7113'
        # where splitting by '.' and converting to int fails on 'post7113'
        version_parts = re.findall(r'\d+', __version__)
        if len(version_parts) >= 3:
            __version_tuple__ = tuple(int(x) for x in version_parts[:3])
        elif len(version_parts) > 0:
            # Pad with zeros if fewer than 3 parts
            __version_tuple__ = tuple(int(x) for x in (version_parts + ['0', '0', '0'])[:3])
        else:
            __version_tuple__ = (0, 0, 0)
    except ImportError:
        logging.getLogger("brian2").warning(
            "Cannot determine Brian version, running from source and "
            "setuptools_scm is not installed."
        )
        __version__ = "unknown"
        __version_tuple__ = (0, 0, 0)

# delete some annoying names from the namespace
if "x" in globals():
    del x
if "f" in globals():
    del f
if "rate" in globals():
    del rate

__docformat__ = "restructuredtext en"

from brian2.only import *
from brian2.only import test

# Initialize the logging system
BrianLogger.initialize()
logger = get_logger(__name__)

# Check the caches
def _get_size_recursively(dirname):
    import os

    total_size = 0
    for dirpath, _, filenames in os.walk(dirname):
        for fname in filenames:
            try:
                size = os.path.getsize(os.path.join(dirpath, fname))
                total_size += size
            except OSError:
                pass 
    return total_size

#: Stores the cache directory for code generation targets
_cache_dirs_and_extensions = {}

#: Flag to track if caches have been initialized
_caches_checked = False

def _check_caches_impl():
    """Internal implementation for checking caches (deferred import of Cython)."""
    from brian2.codegen.runtime.cython_rt.extension_manager import (
        get_cython_cache_dir,
        get_cython_extensions,
    )

    for target, (dirname, extensions) in [
        ("cython", (get_cython_cache_dir(), get_cython_extensions()))
    ]:
        _cache_dirs_and_extensions[target] = (dirname, extensions)
        if prefs.codegen.max_cache_dir_size > 0:
            _check_cache_size(target)

def _ensure_caches_checked():
    """Lazy initialization: checks caches on first use."""
    global _caches_checked
    if not _caches_checked:
        _caches_checked = True
        _check_caches_impl()

def _check_cache_size(target):
    """Public function to check cache size (lazy-loads cache info)."""
    _ensure_caches_checked()
    cache_dir, _ = _cache_dirs_and_extensions.get(target, (None, None))
    if cache_dir is None:
        return
    size = _get_size_recursively(cache_dir)
    size_in_mb = int(round(size / 1024.0 / 1024.0))
    if size_in_mb > prefs.codegen.max_cache_dir_size:
        logger.info(
            f"Cache size for target '{target}': {size_in_mb} MB.\n"
            f"You can call clear_cache('{target}') to delete all "
            "files from the cache or manually delete files in the "
            f"'{cache_dir}' directory."
        )
    else:
        logger.debug(f"Cache size for target '{target}': {size_in_mb} MB")

def check_cache(target):
    """Check the size of the cache for a particular code generation target."""
    _check_cache_size(target)

def clear_cache(target):
    """Clear the cache for a particular code generation target."""
    import os
    import shutil

    _ensure_caches_checked()
    cache_dir, extensions = _cache_dirs_and_extensions.get(target, (None, None))
    if cache_dir is None:
        raise ValueError(f'No cache directory registered for target "{target}".')
    cache_dir = os.path.abspath(cache_dir)
    for folder, _, files in os.walk(cache_dir):
        for f in files:
            for ext in extensions:
                if f.endswith(ext):
                    break
            else:
                raise OSError(
                    f"The cache directory for target '{target}' contains "
                    f"the file '{os.path.join(folder, f)}' of an unexpected type."
                )
    if os.path.exists(cache_dir):
        logger.debug(f"Clearing cache for target '{target}' (directory '{cache_dir}').")
        shutil.rmtree(cache_dir)

# FIXED: _check_caches() is now deferred and executed lazily on first use
# This prevents Cython and pytest imports at startup (see issue #1770) 

class _InterruptHandler:
    def __init__(self, previous_handler):
        self.previous_handler = previous_handler

    def __call__(self, signalnum, stack_frame):
        if (
            not prefs.core.stop_on_keyboard_interrupt
            or not Network._globally_running
            or Network._globally_stopped
        ):
            self.previous_handler(signalnum, stack_frame)
        else:
            logging.getLogger("brian2").warning(
                "Simulation stop requested. Press Ctrl+C again to interrupt."
            )
            Network._globally_stopped = True

_int_handler = _InterruptHandler(signal.getsignal(signal.SIGINT))
signal.signal(signal.SIGINT, _int_handler)

# FIXED: Lazy loading for pylab (matplotlib) to avoid slow startup
# Users can access pylab via: from brian2 import pylab or brian2.pylab
def __getattr__(name):
    """Lazy loading of heavy dependencies like pylab.
    
    This allows users to do: 'from brian2 import pylab' without
    loading matplotlib at startup. Matplotlib will only be loaded
    when the user actually accesses pylab.
    
    Follows SPEC 0001 for lazy searching/importing.
    """
    if name == 'pylab':
        # Lazy load matplotlib.pylab only when accessed
        import matplotlib.pylab as pylab
        globals()['pylab'] = pylab
        return pylab
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")