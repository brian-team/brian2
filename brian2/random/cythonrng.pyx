# cython: language_level=3
# distutils: language = c++
# distutils: include_dirs = brian2/devices/cpp_standalone/brianlib

"""
Global random number generator for Brian2's Cython runtime.

This module provides a single global RandomGenerator instance that is shared
by all generated Cython code. This ensures:
1. Consistent random number sequences across all code objects
2. Identical behavior to C++ standalone mode (which also uses a global generator)
3. Proper state save/restore functionality
"""

from libcpp.string cimport string

cdef extern from "randomgenerator.h":
    cdef cppclass RandomGenerator:
        Randomgenerator() except +
        void seed() except +
        void seed(unsigned long) except +
        double rand() noexcept nogil
        double randn() noexcept nogil
        string get_state()
        void set_state(string) except +

# The ONE global random generator instance
# This is created when the module is first imported and lives for the entire process
cdef RandomGenerator _global_rng


#####  C-level functions (for use by generated Cython code via cimport) #####
cdef double _rand() noexcept nogil:
    return _global_rng.rand()

cdef double _randn() noexcept nogil:
    return _global_rng.randn()



##### Python-level functions (for use by Brian2's Python code) #####
def seed(seed_value=None):
    """
    Seed the global random number generator.

    Parameters
    ----------
    seed_value : int or None
        If None, seed with system entropy (non-deterministic).
        If an integer, seed with that value (deterministic/reproducible).
    """
    if seed_value is None:
        _global_rng.seed()
    else:
        _global_rng.seed(<unsigned long>seed_value)

def get_state():
    """
    Get the complete internal state of the random generator.

    Returns a string that captures:
    - The full MT19937 internal state (624 x 32-bit words)
    - The Box-Muller cached value (for randn)
    - Whether there's a cached value

    This allows exact restoration to this point in the sequence.

    Returns
    -------
    str
        The serialized state that can be passed to set_state().
    """
    cdef string state = _global_rng.get_state()
    return state.decode('utf-8')

def set_state(state_str):
    """
    Restore the random generator to a previously saved state.

    Parameters
    ----------
    state_str : str
        A state string previously returned by get_state().
    """
    cdef bytes state_bytes = state_str.encode('utf-8')
    cdef string state = <string> state_bytes
    _global_rng.set_state(state)


def rand():
    return _global_rng.rand()

def randn():
    return _global_rng.randn()
