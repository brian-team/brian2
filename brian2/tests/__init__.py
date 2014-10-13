import os
import sys
from StringIO import StringIO

from brian2.core.preferences import prefs

def run(codegen_targets=None, long_tests=False, test_codegen_independent=True,
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
    long_tests : bool, optional
        Whether to run tests that take a long time. Defaults to ``False``.
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
        sys.stderr.write('for targets %s' % (', '.join(codegen_targets)))
        ex_in = 'including' if long_tests else 'excluding'
        sys.stderr.write(' (%s long tests)\n' % ex_in)
    else:
        sys.stderr.write('\n')
    if test_standalone:
        sys.stderr.write('Testing standalone \n')
    if test_codegen_independent:
        sys.stderr.write('Testing codegen-independent code \n')
    sys.stderr.write('\n')
    # Store the currently set preferences and reset to default preferences
    stored_prefs = prefs.as_file
    prefs.read_preference_file(StringIO(prefs.defaults_as_file))
    stored_prefs = prefs.as_file
    prefs.read_preference_file(StringIO(prefs.defaults_as_file))

    for target in ['cython', 'weave']:
        if target in codegen_targets:
            # Switch off code optimization to get faster compilation times
            prefs['codegen.runtime.%s.extra_compile_args' % target] = ['-w', '-O0']
    try:
        success = []
        if test_codegen_independent:
            sys.stderr.write('Running tests that do not use code generation\n')
            # Some doctests do actually use code generation, use numpy for that
            prefs.codegen.target = 'numpy'
            prefs._backup()
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
            prefs.codegen.target = target
            prefs._backup()
            exclude_str = "!standalone,!codegen-independent"
            if not long_tests:
                exclude_str += ',!long'
            # explicitly ignore the brian2.hears file for testing, otherwise the
            # doctest search will import it, failing on Python 3
            success.append(nose.run(argv=['', dirname,
                                          '-c=',  # no config file loading
                                          '-I', '^hears\.py$',
                                          '-I', '^\.',
                                          '-I', '^_',
                                          # Do not run standalone or
                                          # codegen-independent tests
                                          "-a", exclude_str,
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
        prefs.read_preference_file(StringIO(stored_prefs))
        prefs._backup()

if __name__=='__main__':
    run()
