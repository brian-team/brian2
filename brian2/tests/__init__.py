import os


def run():
    '''
    Run brian's test suite. Needs an installation of the nose testing tool.
    '''
    try:
        import nose
    except ImportError:
        raise ImportError('Running the test suite requires the "nose" package.')
    
    dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print 'Running tests in "%s"' % dirname
    return nose.run(argv=['', dirname, '--with-doctest', '--nologcapture', '--exe'])

if __name__=='__main__':
    run()
