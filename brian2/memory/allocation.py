'''
Memory management
'''

from numpy import zeros, dtype
from brian2.core.preferences import brian_prefs

__all__ = ['allocate_array',
           ]

brian_prefs.define('default_scalar_dtype', float,
    '''
    Default dtype for all arrays of scalars (state variables, weights, etc.).
    ''', validator=dtype)

def allocate_array(shape, dtype=None):
    '''
    Allocates a 1D array initialised to 0
    
    Parameters
    ----------
    shape : (int, tuple)
        The shape of the array.
    dtype : dtype, optional
        The numpy datatype of the array. If not specified, use the
        :bpref:`default_scalar_dtype` preference. 
        
    Returns
    -------
    arr : ndarray
        The allocated array (initialised to zero).
    '''
    if dtype is None:
        dtype = brian_prefs.default_scalar_dtype
    arr = zeros(shape, dtype=dtype)
    return arr

if __name__=='__main__':
    arr = allocate_array(100)
    print arr.shape, arr.dtype
    arr = allocate_array((100, 2), dtype=int)
    print arr.shape, arr.dtype