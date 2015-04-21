import os
import sys
from StringIO import StringIO

from brian2.core.preferences import prefs
from brian2.devices.device import all_devices

def run(codegen_targets=None, long_tests=False, test_codegen_independent=True,
        test_standalone=None):
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
    test_standalone : str, optional
        Whether to run tests for a standalone mode. Should be the name of a
        standalone mode (e.g. ``'cpp_standalone'``) and expects that a device
        of that name and an accordingly named "simple" device (e.g.
        ``'cpp_standalone_simple'`` exists that can be used for testing (see
        `CPPStandaloneSimpleDevice` for details. Defaults to ``None``, meaning
        that no standalone device is tested.
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
            try:
                import weave
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

    if test_standalone:
        if not isinstance(test_standalone, basestring):
            raise ValueError('test_standalone argument has to be the name of a '
                             'standalone device (e.g. "cpp_standalone")')
        if test_standalone not in all_devices:
            raise ValueError('test_standalone argument "%s" is not a known '
                             'device. Known devices are: '
                             '%s' % (test_standalone,
                                     ', '.join(repr(d) for d in all_devices)))
        sys.stderr.write('Testing standalone \n')
    if test_codegen_independent:
        sys.stderr.write('Testing codegen-independent code \n')
    sys.stderr.write('\n')
    # Store the currently set preferences and reset to default preferences
    stored_prefs = prefs.as_file
    prefs.read_preference_file(StringIO(prefs.defaults_as_file))

    # Switch off code optimization to get faster compilation times
    prefs['codegen.runtime.cython.extra_compile_args'] = ['-w', '-O0']
    prefs['codegen.cpp.extra_compile_args_gcc'] = ['-w', '-O0']
    
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
            exclude_str = "!standalone-only,!codegen-independent"
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
            from brian2.devices.device import get_device, set_device
            previous_device = get_device()
            set_device(test_standalone + '_simple')
            sys.stderr.write('Testing standalone device "%s"\n' % test_standalone)
            sys.stderr.write('Running standalone-compatible standard tests\n')
            exclude_str = ',!long' if not long_tests else ''
            success.append(nose.run(argv=['', dirname,
                                          '-c=',  # no config file loading
                                          '-I', '^hears\.py$',
                                          '-I', '^\.',
                                          '-I', '^_',
                                          # Only run standalone tests
                                          '-a', 'standalone-compatible'+exclude_str,
                                          '--nologcapture',
                                          '--exe']))
            set_device(previous_device)
            sys.stderr.write('Running standalone-specific tests\n')
            success.append(nose.run(argv=['', dirname,
                                          '-c=',  # no config file loading
                                          '-I', '^hears\.py$',
                                          '-I', '^\.',
                                          '-I', '^_',
                                          # Only run standalone tests
                                          '-a', test_standalone+exclude_str,
                                          '--nologcapture',
                                          '--exe']))
        all_success = all(success)
        if not all_success:
            sys.stderr.write(('ERROR: %d/%d test suite(s) did not complete '
                              'successfully (see above).\n') % (len(success) - sum(success),
                                                                len(success)))
        else:
            sys.stderr.write(('OK: %d/%d test suite(s) did complete '
                              'successfully.\n') % (len(success), len(success)))
        return all_success

    finally:
        # Restore the user preferences
        prefs.read_preference_file(StringIO(stored_prefs))
        prefs._backup()

if __name__=='__main__':
    run()
