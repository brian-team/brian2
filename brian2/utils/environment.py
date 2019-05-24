from __future__ import absolute_import
'''
Utility functions to get information about the environment Brian is running in.
'''
from future import builtins


def running_from_ipython():
    '''
    Check whether we are currently running under ipython.
    
    Returns
    -------
    ipython : bool
        Whether running under ipython or not.
    '''
    return getattr(builtins, '__IPYTHON__', False)
