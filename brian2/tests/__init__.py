import os
import sys
from StringIO import StringIO

from brian2.core.preferences import brian_prefs

def run(codegen_targets=None, test_codegen_independent=True,
        test_standalone=False):
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
    test_codegen_independent : bool, optional
        Whether to run tests that are independent of code generation. Defaults
        to ``True``.
    test_standalone : bool, optional
        Whether to run tests for the C++ standalone mode. Defaults to ``False``.
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
    sys.stderr.write('Running tests in "%s" ' % dirname)
    if codegen_targets:
        sys.stderr.write('for targets %s\n' % (', '.join(codegen_targets)))
    else:
        sys.stderr.write('\n')
    if test_standalone:
        sys.stderr.write('Testing standalone \n')
    if test_codegen_independent:
        sys.stderr.write('Testing codegen-independent code \n')
    sys.stderr.write('\n')
    # Store the currently set preferences and reset to default preferences
    stored_prefs = brian_prefs.as_file
    brian_prefs.read_preference_file(StringIO(brian_prefs.defaults_as_file))
    # Switch off code optimization to get faster compilation times
    brian_prefs['codegen.runtime.weave.extra_compile_args'] = ['-w', '-O0']
    brian_prefs['codegen.runtime.cython.extra_compile_args'] = ['-w', '-O0']
    try:
        success = []
        if test_codegen_independent:
            sys.stderr.write('Running tests that do not use code generation\n')
            # Some doctests do actually use code generation, use numpy for that
            brian_prefs.codegen.target = 'numpy'
            brian_prefs._backup()
            success.append(nose.run(argv=['', dirname,
                              '-c=',  # no config file loading
                              '-I', '^hears\.py$',
                              '-I', '^\.',
                              '-I', '^_',
                              '--with-doctest',
                              "-a", "codegen-independent",
                              '--nologcapture',
                              '--exe']))
        for target in codegen_targets:
            sys.stderr.write('Running tests for target %s:\n' % target)
            brian_prefs.codegen.target = target
            brian_prefs._backup()
            # explicitly ignore the brian2.hears file for testing, otherwise the
            # doctest search will import it, failing on Python 3
            success.append(nose.run(argv=['', dirname,
                                          '-c=',  # no config file loading
                                          '-I', '^hears\.py$',
                                          '-I', '^\.',
                                          '-I', '^_',
                                          # Do not run standalone or
                                          # codegen-independent tests
                                          "-a", "!standalone,!codegen-independent",
                                          '--nologcapture',
                                          '--exe']))
        if test_standalone:
            sys.stderr.write('Running standalone tests\n')
            success.append(nose.run(argv=['', dirname,
                                          '-c=',  # no config file loading
                                          '-I', '^hears\.py$',
                                          '-I', '^\.',
                                          '-I', '^_',
                                          # Only run standalone tests
                                          '-a', 'standalone',
                                          '--nologcapture',
                                          '--exe']))
        return all(success)

    finally:
        # Restore the user preferences
        brian_prefs.read_preference_file(StringIO(stored_prefs))
        brian_prefs._backup()

if __name__=='__main__':
    run()
