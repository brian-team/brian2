import pytest

from brian2 import NeuronGroup, prefs
from brian2.codegen.runtime.cython_rt import CythonCodeObject
from brian2.core.preferences import PreferenceError


@pytest.fixture
def cython_available():
    if not CythonCodeObject.is_available():
        pytest.skip("Cython code generation is not available")


def _get_run_code():
    group = NeuronGroup(
        2,
        "dv/dt = -v / (10*ms) : 1",
        codeobj_class=CythonCodeObject,
    )
    group.state_updater.create_code_objects({})
    return group.state_updater.code_objects[0].code.run


def test_cython_directive_defaults(cython_available):
    code = _get_run_code()
    assert "#cython: language_level=3" in code
    assert "#cython: boundscheck=False" in code
    assert "#cython: cdivision=False" in code


def test_cython_directive_overwrite(cython_available):
    old_directives = prefs.codegen.runtime.cython.overwrite_directives
    try:
        prefs.codegen.runtime.cython.overwrite_directives = {
            "cdivision": True,
            "boundscheck": True,
        }
        code = _get_run_code()
        assert "#cython: cdivision=True" in code
        assert "#cython: boundscheck=True" in code
        # Directives that were not overwritten keep their default value
        assert "#cython: wraparound=False" in code
    finally:
        prefs.codegen.runtime.cython.overwrite_directives = old_directives


def test_cython_directive_string_value(cython_available):
    old_directives = prefs.codegen.runtime.cython.overwrite_directives
    try:
        prefs.codegen.runtime.cython.overwrite_directives = {"language_level": "3str"}
        code = _get_run_code()
        assert "#cython: language_level=3str" in code
        assert "#cython: language_level='3str'" not in code
    finally:
        prefs.codegen.runtime.cython.overwrite_directives = old_directives


def test_cython_directive_overwrite_validation():
    old_directives = prefs.codegen.runtime.cython.overwrite_directives
    try:
        prefs.codegen.runtime.cython.overwrite_directives = {"cdivision": True}
        assert prefs.codegen.runtime.cython.overwrite_directives == {"cdivision": True}
    finally:
        prefs.codegen.runtime.cython.overwrite_directives = old_directives

    with pytest.raises(PreferenceError):
        prefs.codegen.runtime.cython.overwrite_directives = ["cdivision"]
