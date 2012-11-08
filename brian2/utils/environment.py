'''
Utility functions to get information about the environment Brian is running in.
'''

def running_from_ipython():
    '''
    Check whether we are currently running under ipython.
    
    Returns
    -------
    ipython : bool
        Whether running under ipython or not.
    '''
    try:
        __IPYTHON__  # @UndefinedVariable 
        return True
    except NameError:
        return False