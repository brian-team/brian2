'''
Utility functions to get information about the environment Brian is running in.
'''

from __future__ import absolute_import

try:
    # Python 2
    import __builtin__ as builtins
except ImportError:
    # Python 3
    import builtins

def running_from_ipython():
    '''
    Check whether we are currently running under ipython.
    
    Returns
    -------
    ipython : bool
        Whether running under ipython or not.
    '''
    return getattr(builtins, '__IPYTHON__', False)
