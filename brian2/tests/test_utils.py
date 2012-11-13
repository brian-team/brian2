from brian2.utils.environment import running_from_ipython

def test_environment():
    '''
    Test information about the environment we are running under.
    '''
    import __builtin__
    
    if hasattr(__builtin__, '__IPYTHON__'):
        testing_under_ipython = True
        del __builtin__.__IPYTHON__
    else:
        testing_under_ipython = False
    
    assert not running_from_ipython()
    __builtin__.__IPYTHON__ = True
    assert running_from_ipython()
    
    if not testing_under_ipython:
        del __builtin__.__IPYTHON__


if __name__ == '__main__':
    test_environment()

    