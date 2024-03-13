"""
Package contain all unit/integration tests for the `brian2` package.
"""

import os
import sys
import tempfile
from io import StringIO

import numpy as np

import brian2
from brian2.core.preferences import prefs
from brian2.devices.device import all_devices, reinit_and_delete, reset_device

try:
    import importlib

    import pytest
    from _pytest import doctest as pytest_doctest

    class OurDoctestModule(pytest_doctest.DoctestModule):
        def collect(self):
            for item in super().collect():
                # Check the object for exclusion from doctests
                full_name = item.name.split(".")
                test_name = []
                module_name = os.path.splitext(os.path.basename(self.name))[0]
                while full_name[-1] != module_name:
                    test_name.append(full_name.pop())
                tested_obj = self.obj
                for name in reversed(test_name):
                    tested_obj = getattr(tested_obj, name)
                if not getattr(tested_obj, "_do_not_run_doctests", False):
                    yield item

    # Monkey patch pytest
    pytest_doctest.DoctestModule = OurDoctestModule

except ImportError:
    pytest = None


class PreferencePlugin:
    def __init__(self, prefs, fail_for_not_implemented=True):
        self.prefs = prefs
        self.device = "runtime"
        self.device_options = {}
        self.fail_for_not_implemented = fail_for_not_implemented

    def pytest_configure(self, config):
        config.brian_prefs = dict(self.prefs)
        config.fail_for_not_implemented = self.fail_for_not_implemented
        config.device = self.device
        config.device_options = self.device_options
        if config.pluginmanager.hasplugin("xdist"):
            xdist_plugin = XDistPreferencePlugin(self)
            config.pluginmanager.register(xdist_plugin)


class XDistPreferencePlugin:
    def __init__(self, pref_plugin):
        self._pref_plugin = pref_plugin

    def pytest_configure_node(self, node):
        """xdist hook"""
        prefs = dict(self._pref_plugin.prefs)
        for k, v in prefs.items():
            if isinstance(v, type):
                prefs[k] = ("TYPE", repr(v))
        node.workerinput["brian_prefs"] = prefs
        node.workerinput["fail_for_not_implemented"] = (
            self._pref_plugin.fail_for_not_implemented
        )
        node.workerinput["device"] = self._pref_plugin.device
        node.workerinput["device_options"] = self._pref_plugin.device_options


def clear_caches():
    from brian2.utils.logger import BrianLogger

    BrianLogger._log_messages.clear()
    from brian2.codegen.translation import make_statements

    make_statements._cache.clear()


def make_argv(dirnames, markers=None, doctests=False, test_GSL=False):
    """
    Create the list of arguments for the ``pytests`` call.

    Parameters
    ----------
    markers : str, optional
        The markers of the tests to include.
    doctests : bool, optional
        Whether to run doctests. Defaults to ``False``.
    test_GSL : bool, optional
        Whether to run tests requiring the GSL. If set to
        ``False``, tests marked with ``gsl`` will be
        excluded. Defaults to ``False``.

    Returns
    -------
    argv : list of str
        The arguments for `pytest.main`.

    """
    if doctests:
        if markers is not None:
            raise TypeError("Cannot give markers for doctests")
        argv = dirnames + [
            "-c",
            os.path.join(os.path.dirname(__file__), "pytest.ini"),
            "--quiet",
            "--doctest-modules",
            "--doctest-glob=*.rst",
            "--doctest-ignore-import-errors",
            "--confcutdir",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            "--pyargs",
            "brian2",
        ]
        if len(dirnames) == 2:
            # If we are testing files in docs_sphinx, ignore conf.py
            argv += [f"--ignore={os.path.join(dirnames[1], 'conf.py')}"]
    else:
        if not test_GSL:
            markers += " and not gsl"
        argv = dirnames + [
            "-c",
            os.path.join(os.path.dirname(__file__), "pytest.ini"),
            "--quiet",
            "-m",
            f"{markers}",
            "--confcutdir",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        ]
    return argv


def run(
    codegen_targets=None,
    long_tests=False,
    test_codegen_independent=True,
    test_standalone=None,
    test_openmp=False,
    test_in_parallel=["codegen_independent", "numpy", "cython", "cpp_standalone"],
    reset_preferences=True,
    fail_for_not_implemented=True,
    test_GSL=False,
    build_options=None,
    extra_test_dirs=None,
    sphinx_dir=None,
    float_dtype=None,
    additional_args=None,
):
    """
    Run brian's test suite. Needs an installation of the pytest testing tool.

    For testing, the preferences will be reset to the default preferences.
    After testing, the user preferences will be restored.

    Parameters
    ----------
    codegen_targets : list of str or str
        A list of codegeneration targets or a single target, e.g.
        ``['numpy', 'cython']`` to test. The whole test suite will be repeatedly
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
    test_GSL : bool, optional
        Whether to test support for GSL state updaters (requires an installation
        of the GSL development packages). Defaults to ``False``.
    build_options : dict, optional
        Non-default build options that will be passed as arguments to the
        `set_device` call for the device specified in ``test_standalone``.
    extra_test_dirs : list of str or str, optional
        Additional directories as a list of strings (or a single directory as
        a string) that will be searched for additional tests.
    sphinx_dir : str, optional
        The full path to ``docs_sphinx``, in order to execute doc tests in the
        rst files. If not provided, assumes we are running from a checked out
        repository where the directory can be found at ``../../docs_sphinx``.
        Will ignore the provided directory if it does not exist.
    float_dtype : np.dtype, optional
        Set the dtype to use for floating point variables to a value different
        from the default `core.default_float_dtype` setting.
    additional_args : list of str, optional
        Optional command line arguments to pass to ``pytest``
    """
    if pytest is None:
        raise ImportError("Running the test suite requires the 'pytest' package.")

    if build_options is None:
        build_options = {}

    if os.name == "nt":
        test_in_parallel = []

    if extra_test_dirs is None:
        extra_test_dirs = []
    elif isinstance(extra_test_dirs, str):
        extra_test_dirs = [extra_test_dirs]
    if additional_args is None:
        additional_args = []

    multiprocess_arguments = ["-n", "auto"]

    if codegen_targets is None:
        codegen_targets = ["numpy"]
        try:
            import Cython

            codegen_targets.append("cython")
        except ImportError:
            pass
    elif isinstance(codegen_targets, str):  # allow to give a single target
        codegen_targets = [codegen_targets]

    dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dirnames = [dirname] + extra_test_dirs

    print(f"Running tests in {', '.join(dirnames)} ", end="")
    if codegen_targets:
        print(f"for targets {', '.join(codegen_targets)}", end="")
    ex_in = "including" if long_tests else "excluding"
    print(f" ({ex_in} long tests)")

    print(
        f"Running Brian version {brian2.__version__} from"
        f" '{os.path.dirname(brian2.__file__)}'"
    )

    all_targets = set(codegen_targets)

    if test_standalone:
        if not isinstance(test_standalone, str):
            raise ValueError(
                "test_standalone argument has to be the name of a "
                "standalone device (e.g. 'cpp_standalone')"
            )
        if test_standalone not in all_devices:
            known_devices = ", ".join(repr(d) for d in all_devices)
            raise ValueError(
                "test_standalone argument 'test_standalone' is not a known "
                f"device. Known devices are: {known_devices}."
            )
        print("Testing standalone")
        all_targets.add(test_standalone)
    if test_codegen_independent:
        print("Testing codegen-independent code")
        all_targets.add("codegen_independent")

    parallel_tests = all_targets.intersection(set(test_in_parallel))
    if parallel_tests:
        try:
            import xdist

            print(f"Testing with multiple processes for {', '.join(parallel_tests)}")
        except ImportError:
            test_in_parallel = []

    if reset_preferences:
        print("Resetting to default preferences")
        stored_prefs = prefs.as_file
        prefs.reset_to_defaults()

    # Avoid failures in the tests for user-registered units
    import copy

    import brian2.units.fundamentalunits as fundamentalunits

    old_unit_registry = copy.copy(fundamentalunits.user_unit_register)
    fundamentalunits.user_unit_register = fundamentalunits.UnitRegistry()

    if float_dtype is not None:
        print(f"Setting dtype for floating point variables to: {float_dtype.__name__}")

        prefs["core.default_float_dtype"] = float_dtype

    print()

    # Suppress INFO log messages during testing
    from brian2.utils.logger import LOG_LEVELS, BrianLogger

    log_level = BrianLogger.console_handler.level
    BrianLogger.console_handler.setLevel(LOG_LEVELS["WARNING"])

    # Switch off code optimization to get faster compilation times
    prefs["codegen.cpp.extra_compile_args_gcc"].extend(["-w", "-O0"])
    prefs["codegen.cpp.extra_compile_args_msvc"].extend(["/Od"])

    pref_plugin = PreferencePlugin(prefs, fail_for_not_implemented)
    try:
        success = []
        pref_plugin.device = "runtime"
        pref_plugin.device_options = {}
        if test_codegen_independent:
            print("Running doctests")
            # Some doctests do actually use code generation, use numpy for that
            prefs["codegen.target"] = "numpy"
            # Always test doctests with 64 bit, to avoid differences in print output
            if float_dtype is not None:
                prefs["core.default_float_dtype"] = np.float64
            if sphinx_dir is None:
                sphinx_dir = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..", "docs_sphinx")
                )
            if os.path.exists(sphinx_dir):
                sphinx_doc_dir = [sphinx_dir]
            else:
                # When running on travis, the source directory is in the SRCDIR
                # environment variable
                if "SRCDIR" in os.environ:
                    sphinx_dir = os.path.abspath(
                        os.path.join(os.environ["SRCDIR"], "docs_sphinx")
                    )
                    if os.path.exists(sphinx_dir):
                        sphinx_doc_dir = [sphinx_dir]
                    else:
                        sphinx_doc_dir = []
                else:
                    sphinx_doc_dir = []
            argv = make_argv(dirnames + sphinx_doc_dir, doctests=True)
            if "codegen_independent" in test_in_parallel:
                argv.extend(multiprocess_arguments)
            success.append(
                pytest.main(argv + additional_args, plugins=[pref_plugin]) == 0
            )
            # Set float_dtype back again if necessary
            if float_dtype is not None:
                prefs["core.default_float_dtype"] = float_dtype

            print("Running tests that do not use code generation")
            argv = make_argv(dirnames, "codegen_independent", test_GSL=test_GSL)
            if "codegen_independent" in test_in_parallel:
                argv.extend(multiprocess_arguments)
            success.append(
                pytest.main(argv + additional_args, plugins=[pref_plugin]) == 0
            )
            clear_caches()

        for target in codegen_targets:
            print(f"Running tests for target {target}:")
            # Also set the target for string-expressions -- otherwise we'd only
            # ever test numpy for those
            prefs["codegen.target"] = target

            markers = "not standalone_only and not codegen_independent"
            if not long_tests:
                markers += " and not long"
            # explicitly ignore the brian2.hears file for testing, otherwise the
            # doctest search will import it, failing on Python 3
            argv = make_argv(dirnames, markers, test_GSL=test_GSL)
            if target in test_in_parallel:
                argv.extend(multiprocess_arguments)
            success.append(
                pytest.main(argv + additional_args, plugins=[pref_plugin]) == 0
            )
            clear_caches()

        pref_plugin.device = test_standalone
        if test_standalone:
            from brian2.devices.device import get_device, set_device

            pref_plugin.device_options = {"directory": None, "with_output": False}
            pref_plugin.device_options.update(build_options)
            print(f'Testing standalone device "{test_standalone}"')
            print("Running standalone-compatible standard tests (single run statement)")
            markers = "and not long" if not long_tests else ""
            markers += " and not multiple_runs"
            argv = make_argv(
                dirnames, f"standalone_compatible {markers}", test_GSL=test_GSL
            )
            if test_standalone in test_in_parallel:
                argv.extend(multiprocess_arguments)
            success.append(
                pytest.main(argv + additional_args, plugins=[pref_plugin]) in [0, 5]
            )
            clear_caches()

            reset_device()

            print(
                "Running standalone-compatible standard tests (multiple run statements)"
            )
            pref_plugin.device_options = {
                "directory": None,
                "with_output": False,
                "build_on_run": False,
            }
            pref_plugin.device_options.update(build_options)
            markers = " and not long" if not long_tests else ""
            markers += " and multiple_runs"
            argv = make_argv(
                dirnames, f"standalone_compatible{markers}", test_GSL=test_GSL
            )
            if test_standalone in test_in_parallel:
                argv.extend(multiprocess_arguments)
            success.append(
                pytest.main(argv + additional_args, plugins=[pref_plugin]) in [0, 5]
            )
            clear_caches()
            reset_device()

            if test_openmp and test_standalone == "cpp_standalone":
                # Run all the standalone compatible tests again with 4 threads
                pref_plugin.device_options = {"directory": None, "with_output": False}
                pref_plugin.device_options.update(build_options)
                prefs["devices.cpp_standalone.openmp_threads"] = 4
                print(
                    "Running standalone-compatible standard tests with OpenMP (single"
                    " run statements)"
                )
                markers = " and not long" if not long_tests else ""
                markers += " and not multiple_runs"
                argv = make_argv(
                    dirnames, f"standalone_compatible{markers}", test_GSL=test_GSL
                )
                success.append(
                    pytest.main(argv + additional_args, plugins=[pref_plugin]) in [0, 5]
                )
                clear_caches()
                reset_device()

                pref_plugin.device_options = {
                    "directory": None,
                    "with_output": False,
                    "build_on_run": False,
                }
                pref_plugin.device_options.update(build_options)
                print(
                    "Running standalone-compatible standard tests with OpenMP (multiple"
                    " run statements)"
                )
                markers = " and not long" if not long_tests else ""
                markers += " and multiple_runs"
                argv = make_argv(
                    dirnames, f"standalone_compatible{markers}", test_GSL=test_GSL
                )
                success.append(
                    pytest.main(argv + additional_args, plugins=[pref_plugin]) in [0, 5]
                )
                clear_caches()
                prefs["devices.cpp_standalone.openmp_threads"] = 0

                reset_device()

            print("Running standalone-specific tests")
            exclude_openmp = " and not openmp" if not test_openmp else ""
            argv = make_argv(
                dirnames, test_standalone + exclude_openmp, test_GSL=test_GSL
            )
            if test_standalone in test_in_parallel:
                argv.extend(multiprocess_arguments)
            success.append(
                pytest.main(argv + additional_args, plugins=[pref_plugin]) in [0, 5]
            )
            clear_caches()

        all_success = all(success)
        if not all_success:
            print(
                f"ERROR: {len(success) - sum(success)}/{len(success)} test suite(s) "
                "did not complete successfully (see above)."
            )
        else:
            print(
                f"OK: {len(success)}/{len(success)} test suite(s) did complete "
                "successfully."
            )
        return all_success

    finally:
        BrianLogger.console_handler.setLevel(log_level)

        if reset_preferences:
            # Restore the user preferences
            prefs.read_preference_file(StringIO(stored_prefs))
            prefs._backup()

        fundamentalunits.user_unit_register = old_unit_registry


if __name__ == "__main__":
    run()
