import sys
import os

from numpy.testing.noseclasses import KnownFailure

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
    return nose.run(argv=['', dirname, '--with-doctest',
                          '--nologcapture', '--exe'],
                    addplugins=[KnownFailure()])


def expected_python3_failure(test):
    import nose.tools
    from numpy.testing.noseclasses import KnownFailureTest
    if sys.version_info[0] == 2:
        return test  # Nothing to do

    @nose.tools.make_decorator(test)
    def inner(*args, **kwargs):
        try:
            test(*args, **kwargs)
        except Exception:
            raise KnownFailureTest('This test is known to not work under Python 3')
        else:
            raise AssertionError('This test was expected to fail under '
                                 'Python 3, but it passed!')
    return inner


if __name__=='__main__':
    run()
