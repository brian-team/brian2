import os
import sys
from StringIO import StringIO

from brian2.core.preferences import prefs
from brian2.devices.device import all_devices, set_device, reset_device

try:
    import nose
    from nose.plugins.errorclass import ErrorClassPlugin, ErrorClass

    class NotImplementedPlugin(ErrorClassPlugin):
        enabled = True
        notimplemented = ErrorClass(NotImplementedError,
                                    label='NOT_IMPLEMENTED',
                                    isfailure=True)

        def configure(self, options, conf):
            # For some reason, this only works if this method exists...
            pass

    class NotImplementedNoFailurePlugin(ErrorClassPlugin):
        enabled = True
        notimplemented = ErrorClass(NotImplementedError,
                                    label='NOT_IMPLEMENTED',
                                    isfailure=False)

        def configure(self, options, conf):
            # For some reason, this only works if this method exists...
            pass

except ImportError:
    nose = None


def run(codegen_targets=None, long_tests=False, test_codegen_independent=True,
        test_standalone=None, test_openmp=False,
        test_in_parallel=['codegen_independent', 'numpy', 'cython', 'cpp_standalone'],
        reset_preferences=True, fail_for_not_implemented=True):
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
    test_openmp : bool, optional
        Whether to test standalone test with multiple threads and OpenMP. Will
        be ignored if ``cpp_standalone`` is not tested. Defaults to ``False``.
    reset_preferences : bool, optional
        Whether to reset all preferences to the default preferences before
        running the test suite. Defaults to ``True`` to get test results
        independent of the user's preference settings but can be switched off
        when the preferences are actually necessary to pass the tests (e.g. for
        device-specific settings).
    fail_for_not_implemented : bool, optional
        Whether to fail for tests raising a `NotImplementedError`. Defaults to
        ``True``, but can be switched off for devices known to not implement
        all of Brian's features.
    '''
    if nose is None:
        raise ImportError('Running the test suite requires the "nose" package.')
    
    if os.name == 'nt':
        test_in_parallel = []

    multiprocess_arguments = ['--processes=-1',
                              '--process-timeout=3600',  # we don't want them to time out
                              '--process-restartworker']

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

    all_targets = set(codegen_targets)

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
        all_targets.add(test_standalone)
    if test_codegen_independent:
        sys.stderr.write('Testing codegen-independent code \n')
        all_targets.add('codegen_independent')

    parallel_tests = all_targets.intersection(set(test_in_parallel))
    if parallel_tests:
        sys.stderr.write('Testing with multiple processes for %s\n' % ', '.join(parallel_tests))

    if reset_preferences:
        sys.stderr.write('Resetting to default preferences\n')

    sys.stderr.write('\n')

    if reset_preferences:
        # Store the currently set preferences and reset to default preferences
        stored_prefs = prefs.as_file
        prefs.read_preference_file(StringIO(prefs.defaults_as_file))

    # Suppress INFO log messages during testing
    from brian2.utils.logger import CONSOLE_HANDLER, LOG_LEVELS
    log_level = CONSOLE_HANDLER.level
    CONSOLE_HANDLER.setLevel(LOG_LEVELS['WARNING'])

    # Switch off code optimization to get faster compilation times
    prefs['codegen.cpp.extra_compile_args_gcc'].extend(['-w', '-O0'])
    prefs['codegen.cpp.extra_compile_args_msvc'].extend(['/Od'])

    if fail_for_not_implemented:
        not_implemented_plugin = NotImplementedPlugin
    else:
        not_implemented_plugin = NotImplementedNoFailurePlugin
    # This hack is needed to get the NotImplementedPlugin working for multiprocessing
    import nose.plugins.multiprocess as multiprocess
    multiprocess._instantiate_plugins = [not_implemented_plugin]

    plugins = [not_implemented_plugin()]

    try:
        success = []
        if test_codegen_independent:
            sys.stderr.write('Running tests that do not use code generation\n')
            # Some doctests do actually use code generation, use numpy for that
            prefs.codegen.target = 'numpy'
            prefs._backup()
            argv = ['nosetests', dirname,
                    '-c=',  # no config file loading
                    '-I', '^hears\.py$',
                    '-I', '^\.',
                    '-I', '^_',
                    '--with-doctest',
                    "-a", "codegen-independent",
                    '--nologcapture',
                    '--exe']
            if 'codegen_independent' in test_in_parallel:
                argv.extend(multiprocess_arguments)
            success.append(nose.run(argv=argv,
                                    addplugins=plugins))

        for target in codegen_targets:
            sys.stderr.write('Running tests for target %s:\n' % target)
            prefs.codegen.target = target
            # Also set the target for string-expressions -- otherwise we'd only
            # ever test numpy for those
            prefs.codegen.string_expression_target = target
            prefs._backup()
            exclude_str = "!standalone-only,!codegen-independent"
            if not long_tests:
                exclude_str += ',!long'
            # explicitly ignore the brian2.hears file for testing, otherwise the
            # doctest search will import it, failing on Python 3
            argv = ['nosetests', dirname,
                    '-c=',  # no config file loading
                    '-I', '^hears\.py$',
                    '-I', '^\.',
                    '-I', '^_',
                    # Do not run standalone or
                    # codegen-independent tests
                    "-a", exclude_str,
                    '--nologcapture',
                    '--exe']
            if target in test_in_parallel:
                argv.extend(multiprocess_arguments)
            success.append(nose.run(argv=argv,
                                    addplugins=plugins))

        if test_standalone:
            from brian2.devices.device import get_device, set_device
            set_device(test_standalone, directory=None,  # use temp directory
                       with_output=False)
            sys.stderr.write('Testing standalone device "%s"\n' % test_standalone)
            sys.stderr.write('Running standalone-compatible standard tests\n')
            exclude_str = ',!long' if not long_tests else ''
            argv = ['nosetests', dirname,
                    '-c=',  # no config file loading
                    '-I', '^hears\.py$',
                    '-I', '^\.',
                    '-I', '^_',
                    # Only run standalone tests
                    '-a', 'standalone-compatible'+exclude_str,
                    '--nologcapture',
                    '--exe']
            if test_standalone in test_in_parallel:
                argv.extend(multiprocess_arguments)
            success.append(nose.run(argv=argv,
                                    addplugins=plugins))

            if test_openmp and test_standalone == 'cpp_standalone':
                # Run all the standalone compatible tests again with 4 threads
                prefs.devices.cpp_standalone.openmp_threads = 4
                prefs._backup()
                sys.stderr.write('Running standalone-compatible standard tests with OpenMP\n')
                exclude_str = ',!long' if not long_tests else ''
                argv = ['nosetests', dirname,
                        '-c=',  # no config file loading
                        '-I', '^hears\.py$',
                        '-I', '^\.',
                        '-I', '^_',
                        # Only run standalone tests
                        '-a', 'standalone-compatible'+exclude_str,
                        '--nologcapture',
                        '--exe']
                success.append(nose.run(argv=argv,
                                        addplugins=plugins))
                prefs.devices.cpp_standalone.openmp_threads = 0
                prefs._backup()

            reset_device()

            sys.stderr.write('Running standalone-specific tests\n')
            exclude_openmp = ',!openmp' if not test_openmp else ''
            argv = ['nosetests', dirname,
                    '-c=',  # no config file loading
                    '-I', '^hears\.py$',
                    '-I', '^\.',
                    '-I', '^_',
                    # Only run standalone tests
                    '-a', test_standalone+exclude_openmp,
                    '--nologcapture',
                    '--exe']
            if test_standalone in test_in_parallel:
                argv.extend(multiprocess_arguments)
            success.append(nose.run(argv=argv,
                                    addplugins=plugins))
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
        CONSOLE_HANDLER.setLevel(log_level)
        if reset_preferences:
            # Restore the user preferences
            prefs.read_preference_file(StringIO(stored_prefs))
            prefs._backup()

if __name__=='__main__':
    run()
