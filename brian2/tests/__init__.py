import os

def run():
    '''
    Run brian's test suite. Needs an installation of the nose testing tool.
    '''
    import nose
    from nose.plugins.doctests import Doctest
    
    dirname = os.path.join(os.path.dirname(__file__), '..')
    return nose.run(argv=['', dirname, '--with-doctest'])

if __name__=='__main__':
    run()
    