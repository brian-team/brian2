import os
import sys
from StringIO import StringIO

from brian2.core.preferences import brian_prefs

def run(codegen_targets=None):
    '''
    Run brian's test suite. Needs an installation of the nose testing tool.

    For testing, the preferences will be reset to the default preferences.
    After testing, the user preferences will be restored.

    Parameters
    ----------
    codegen_targets : list of str or str
        A list of codegeneration targets or a single target, e.g.
        ``['numpy', 'weave']`` to test. The whole test suite will be repeatedly
        run with `codegen.target` set to the respective value. If not
        specified, all available code generation targets will be tested.
    '''
    try:
        import nose
    except ImportError:
        raise ImportError('Running the test suite requires the "nose" package.')

    if codegen_targets is None:
        codegen_targets = ['numpy']
        try:
            import scipy.weave
            codegen_targets.append('weave')
        except ImportError:
            pass
        try:
            import Cython
            codegen_targets.append('cython')
        except ImportError:
            pass
    elif isinstance(codegen_targets, basestring):  # allow to give a single target
        codegen_targets = [codegen_targets]

    dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # We write to stderr since nose does all of its output on stderr as well
    sys.stderr.write('Running tests in "%s" for targets %s\n' % (dirname,
                                                                 ', '.join(codegen_targets)))

    # Store the currently set preferences and reset to default preferences
    stored_prefs = brian_prefs.as_file
    brian_prefs.read_preference_file(StringIO(brian_prefs.defaults_as_file))
    try:
        success = []
        for target in codegen_targets:
            sys.stderr.write('Testing target %s:\n' % target)
            brian_prefs.codegen.target = target
            brian_prefs._backup()
            # explicitly ignore the brian2.hears file for testing, otherwise the
            # doctest search will import it, failing on Python 3
            success.append(nose.run(argv=['', dirname,
                                          '-I', '^hears\.py$',
                                          '-I', '^\.',
                                          '-I', '^_',
                                          '--with-doctest', '--nologcapture', '--exe']))
        return all(success)
    finally:
        # Restore the user preferences
        brian_prefs.read_preference_file(StringIO(stored_prefs))
        brian_prefs._backup()

if __name__=='__main__':
    run()
