from brian2.utils.environment import running_from_ipython

def test_environment():
    '''
    Test information about the environment we are running under.
    '''
    try:
        # Python 2
        import __builtin__ as builtins
    except ImportError:
        # Python 3
        import builtins
    
    if hasattr(builtins, '__IPYTHON__'):
        testing_under_ipython = True
        del builtins.__IPYTHON__
    else:
        testing_under_ipython = False
    
    assert not running_from_ipython()
    builtins.__IPYTHON__ = True
    assert running_from_ipython()
    
    if not testing_under_ipython:
        del builtins.__IPYTHON__


if __name__ == '__main__':
    test_environment()

    