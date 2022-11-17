from brian2 import *
from brian2.codegen.templates import Templater
import pytest


@pytest.mark.codegen_independent
def test_templates():
    T1 = Templater(
        "brian2.tests.test_templates.fake_package_1",
        env_globals={"f": lambda: "F1", "g": lambda: "G1"},
        extension=".txt",
    )
    T2 = T1.derive(
        "brian2.tests.test_templates.fake_package_2", env_globals={"f": lambda: "F2"}
    )
    ns = {}
    for i, T in enumerate([T1, T2], start=1):
        for c in ["A", "B", "C", "D"]:
            ns[c + str(i)] = getattr(T, c)("", "")
    # for k, v in ns.items():
    #     print k, v
    assert "A1" in ns["A1"]
    assert "B1" in ns["A1"]
    assert "F1" in ns["A1"]
    assert "G1" in ns["A1"]
    assert "A2" in ns["A2"]
    assert "F2" in ns["A2"]
    assert "G1" in ns["A2"]
    assert "B1" not in ns["A2"]
    assert "B1" in ns["B1"]
    assert "B1" in ns["B2"]
    assert "C1" in ns["C1"]
    assert "D1" in ns["C1"]
    assert "C1" in ns["C2"]
    assert "D2" in ns["C2"]
    assert "D1" not in ns["C2"]
    assert "D1" in ns["D1"]
    assert "D2" in ns["D2"]


if __name__ == "__main__":
    test_templates()
