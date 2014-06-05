import os
from StringIO import StringIO

from brian2.core.preferences import brian_prefs

def run():
    '''
    Run brian's test suite. Needs an installation of the nose testing tool.

    For testing, the preferences will be reset to the default preferences.
    After testing, the user preferences will be restored.
    '''
    try:
        import nose
    except ImportError:
        raise ImportError('Running the test suite requires the "nose" package.')
    
    dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print 'Running tests in "%s"' % dirname

    # Store the currently set preferences and reset to default preferences
    stored_prefs = brian_prefs.as_file
    brian_prefs.read_preference_file(StringIO(brian_prefs.defaults_as_file))
    try:
        return nose.run(argv=['', dirname, '--with-doctest', '--nologcapture', '--exe'])
    finally:
        # Restore the user preferences
        brian_prefs.read_preference_file(StringIO(stored_prefs))

if __name__=='__main__':
    run()
