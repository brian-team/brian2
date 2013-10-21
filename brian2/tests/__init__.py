import os

def run():
    '''
    Run brian's test suite. Needs an installation of the nose testing tool.
    '''
    try:
        import nose
    except ImportError:
        raise ImportError('Running the test suite requires the "nose" package.')
    
    dirname = os.path.join(os.path.dirname(__file__), '..')
    return nose.run(argv=['', dirname, '--with-doctest'])

if __name__=='__main__':
    run()
    