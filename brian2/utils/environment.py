'''
Utility functions to get information about the environment Brian is running in.
'''

import __builtin__

def running_from_ipython():
    '''
    Check whether we are currently running under ipython.
    
    Returns
    -------
    ipython : bool
        Whether running under ipython or not.
    '''
    return getattr(__builtin__, '__IPYTHON__', False)

