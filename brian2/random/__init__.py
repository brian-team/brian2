# brian2/random/__init__.py
"""
Random number generation for Brian2.

This module provides a unified random number generator that produces
identical sequences in both Cython runtime and C++ standalone modes.
"""

from .cythonrng import seed, get_state, set_state, rand, randn

__all__ = ["seed", "get_state", "set_state", "rand", "randn"]
