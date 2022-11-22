"""
Helper module containing functions that operate on numpy arrays.
"""

import numpy as np


def calc_repeats(delay):
    """
    Calculates offsets corresponding to an array, where repeated values are
    subsequently numbered, i.e. if there n identical values, the returned array
    will have values from 0 to n-1 at their positions.
    The code is complex because tricks are needed for vectorisation.

    This function is used in the Python `SpikeQueue` to calculate the offset
    array for the insertion of spikes with their respective delays into the
    queue and in the numpy code for synapse creation to calculate how many
    synapses for each source-target pair exist.

    Examples
    --------
    >>> import numpy as np
    >>> print(calc_repeats(np.array([7, 5, 7, 3, 7, 5])))
    [0 0 1 0 2 1]
    """
    # We use merge sort because it preserves the input order of equal
    # elements in the sorted output
    I = np.argsort(delay, kind="mergesort")
    xs = delay[I]
    J = xs[1:] != xs[:-1]
    A = np.hstack((0, np.cumsum(J)))
    B = np.hstack((0, np.cumsum(np.logical_not(J))))
    BJ = np.hstack((0, B[:-1][J]))
    ei = B - BJ[A]
    ofs = np.zeros_like(delay, dtype=np.int32)
    ofs[I] = np.array(ei, dtype=ofs.dtype)
    return ofs
