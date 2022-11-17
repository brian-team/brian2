import uuid

import sympy
import numpy
import pytest

from brian2.core.variables import Constant
from brian2.core.namespace import get_local_namespace
from brian2.groups.group import Group
from brian2.units import second, volt
from brian2.units.fundamentalunits import Unit
from brian2.units.stdunits import ms, Hz, mV
from brian2.units.unitsafefunctions import sin, log, exp
from brian2.utils.logger import catch_logs

# a simple Group for testing
class SimpleGroup(Group):
    def __init__(self, variables, namespace=None):
        self.variables = variables
        # We use a unique name to get repeated warnings
        Group.__init__(
            self,
            namespace=namespace,
            name=f"simplegroup_{str(uuid.uuid4()).replace('-', '_')}",
        )


def _assert_one_warning(l):
    assert len(l) == 1, f"expected one warning got {len(l)}"
    assert l[0][0] == "WARNING", f"expected a WARNING, got {l[0][0]} instead"


@pytest.mark.codegen_independent
def test_default_content():
    """
    Test that the default namespace contains standard units and functions.
    """
    group = Group()
    # Units
    assert group._resolve("second", {}).get_value_with_unit() == second
    assert group._resolve("volt", {}).get_value_with_unit() == volt
    assert group._resolve("ms", {}).get_value_with_unit() == ms
    assert group._resolve("Hz", {}).get_value_with_unit() == Hz
    assert group._resolve("mV", {}).get_value_with_unit() == mV

    # Functions
    assert group._resolve("sin", {}).pyfunc == sin
    assert group._resolve("log", {}).pyfunc == log
    assert group._resolve("exp", {}).pyfunc == exp

    # Constants
    assert group._resolve("e", {}).sympy_obj == sympy.E
    assert group._resolve("e", {}).get_value() == numpy.e
    assert group._resolve("pi", {}).sympy_obj == sympy.pi
    assert group._resolve("pi", {}).get_value() == numpy.pi
    assert group._resolve("inf", {}).sympy_obj == sympy.oo
    assert group._resolve("inf", {}).get_value() == numpy.inf


@pytest.mark.codegen_independent
def test_explicit_namespace():
    """Test resolution with an explicitly provided namespace"""
    group = SimpleGroup(namespace={"variable": 42}, variables={})

    # Explicitly provided
    with catch_logs() as l:
        assert group._resolve("variable", {}).get_value_with_unit() == 42
        assert len(l) == 0

    # Value from an explicit run namespace
    with catch_logs() as l:
        assert (
            group._resolve(
                "yet_another_var", run_namespace={"yet_another_var": 17}
            ).get_value_with_unit()
            == 17
        )
        assert len(l) == 0


@pytest.mark.codegen_independent
def test_errors():
    # No explicit namespace
    group = SimpleGroup(namespace=None, variables={})
    with pytest.raises(KeyError):
        group._resolve("nonexisting_variable", {})

    # Empty explicit namespace
    group = SimpleGroup(namespace={}, variables={})
    with pytest.raises(KeyError):
        group._resolve("nonexisting_variable", {})

    # Illegal name
    with pytest.raises(ValueError):
        SimpleGroup(namespace={"_illegal": 3.0}, variables={})


@pytest.mark.codegen_independent
def test_resolution():
    # implicit namespace
    tau = 10 * ms
    group = SimpleGroup(namespace=None, variables={})
    namespace = get_local_namespace(level=0)
    resolved = group.resolve_all(
        ["tau", "ms"], namespace, user_identifiers=["tau", "ms"]
    )
    assert len(resolved) == 2
    assert isinstance(resolved, dict)
    assert resolved["tau"].get_value_with_unit() == tau
    assert resolved["ms"].get_value_with_unit() == ms
    del tau

    # explicit namespace
    group = SimpleGroup(namespace={"tau": 20 * ms}, variables={})
    namespace = get_local_namespace(level=0)
    resolved = group.resolve_all(["tau", "ms"], namespace, ["tau", "ms"])
    assert len(resolved) == 2
    assert resolved["tau"].get_value_with_unit() == 20 * ms


@pytest.mark.codegen_independent
def test_warning():
    from brian2.core.functions import DEFAULT_FUNCTIONS
    from brian2.units.stdunits import cm as brian_cm

    # Name in external namespace clashes with unit/function name
    exp = 23
    cm = 42
    group = SimpleGroup(namespace=None, variables={})
    namespace = get_local_namespace(level=0)
    with catch_logs() as l:
        resolved = group.resolve_all(["exp"], namespace)["exp"]
        assert resolved == DEFAULT_FUNCTIONS["exp"]
        assert len(l) == 1, f"got warnings: {str(l)}"
        assert l[0][1].endswith(".resolution_conflict")
    with catch_logs() as l:
        resolved = group.resolve_all(["cm"], namespace)["cm"]
        assert resolved.get_value_with_unit() == brian_cm
        assert len(l) == 1, f"got warnings: {str(l)}"
        assert l[0][1].endswith(".resolution_conflict")


@pytest.mark.codegen_independent
def test_warning_internal_variables():
    group1 = SimpleGroup(namespace=None, variables={"N": Constant("N", 5)})
    group2 = SimpleGroup(namespace=None, variables={"N": Constant("N", 7)})
    with catch_logs() as l:
        group1.resolve_all(["N"], run_namespace={"N": 5})  # should not raise a warning
        assert len(l) == 0, f"got warnings: {str(l)}"
    with catch_logs() as l:
        group2.resolve_all(["N"], run_namespace={"N": 5})  # should raise a warning
        assert len(l) == 1, f"got warnings: {str(l)}"
        assert l[0][1].endswith(".resolution_conflict")


if __name__ == "__main__":
    test_default_content()
    test_explicit_namespace()
    test_errors()
    test_resolution()
    test_warning()
    test_warning_internal_variables()
