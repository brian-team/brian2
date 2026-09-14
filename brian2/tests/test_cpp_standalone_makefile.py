"""Incremental-build contracts for the GNU make standalone template."""

import os
import shutil
import subprocess
import time
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpp_standalone


def make(folder, *targets, check=True):
    result = subprocess.run(
        [
            "make",
            "-j2",
            "CFLAGS=-std=c11 -DC_SOURCE=1",
            "CPPFLAGS=-DC_SOURCE=2",
            "TARGET_ARCH=-DARCH_FLAG=7",
            *targets,
        ],
        cwd=folder,
        text=True,
        capture_output=True,
        timeout=60,
    )
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def output(folder):
    return subprocess.check_output(
        [str(folder / "main")], cwd=folder, text=True
    ).strip()


def next_timestamp():
    # GNU make 3.81 (shipped with macOS) compares whole-second timestamps.
    time.sleep(1.1)


@pytest.fixture
def make_project(tmp_path):
    from brian2.devices.cpp_standalone.device import CPPStandaloneDevice, CPPWriter

    if os.name == "nt":
        pytest.skip("Requires a POSIX shell and GNU make")
    if any(shutil.which(tool) is None for tool in ("make", "cc", "c++")):
        pytest.skip("Requires make and C/C++ compilers")
    (tmp_path / "nested").mkdir()
    writer = CPPWriter(str(tmp_path))
    writer.write("nested/value.h", "#define VALUE 21\n")
    writer.write(
        "nested/value.cpp", '#include "value.h"\nint cpp_value() { return VALUE; }\n'
    )
    writer.write(
        "nested/value_c.c",
        """#include "value.h"
#if defined(__cplusplus) || C_SOURCE != 2 || ARCH_FLAG != 7
#error C source must preserve the built-in C compilation flags and their order
#endif
int c_value(void) { return VALUE; }
""",
    )
    writer.write(
        "main.cpp",
        """#include <iostream>
int cpp_value();
extern "C" int c_value(void);
int main() { std::cout << cpp_value() + c_value() << std::endl; }
""",
    )
    CPPStandaloneDevice().generate_makefile(writer, "unix", "-O0", "", 0, False)
    return tmp_path


def test_makefile_clean_and_noop(make_project):
    folder = make_project
    make(folder, "clean")  # Cleaning a never-built project is valid.
    make(folder)
    assert output(folder) == "42"
    objects = list(folder.rglob("*.o"))
    assert len(objects) == 3
    for obj in objects:
        assert (
            obj.with_suffix(".d")
            .read_text()
            .startswith(str(obj.relative_to(folder)) + ":")
        )
    before = {obj: obj.stat().st_mtime_ns for obj in objects}
    make(folder)
    assert before == {obj: obj.stat().st_mtime_ns for obj in objects}
    make(folder, "clean")
    make(folder, "clean")
    assert not list(folder.rglob("*.o"))
    assert not list(folder.rglob("*.d"))
    assert not (folder / "main").exists()
    make(folder)
    assert output(folder) == "42"


def test_makefile_nested_header_change(make_project):
    folder = make_project
    make(folder)
    main_time = (folder / "main.o").stat().st_mtime_ns
    next_timestamp()
    (folder / "nested/value.h").write_text("#define VALUE 22\n")
    make(folder)
    assert output(folder) == "44"
    assert (folder / "main.o").stat().st_mtime_ns == main_time


@pytest.mark.parametrize("suffix", ["value", "value_c", "all"])
def test_makefile_missing_dependencies(make_project, suffix):
    folder = make_project
    make(folder)
    deps = (
        list(folder.rglob("*.d"))
        if suffix == "all"
        else [folder / f"nested/{suffix}.d"]
    )
    for dep in deps:
        dep.unlink()
    make(folder)
    assert all(dep.exists() for dep in deps)
    next_timestamp()
    (folder / "nested/value.h").write_text("#define VALUE 23\n")
    make(folder)
    assert output(folder) == "46"


@pytest.mark.parametrize("suffix", ["value", "value_c"])
def test_makefile_missing_dependency_direct_target(make_project, suffix):
    folder = make_project
    make(folder)
    dep = folder / f"nested/{suffix}.d"
    dep.unlink()
    make(folder, f"nested/{suffix}.o")
    assert dep.exists()


def test_makefile_deleted_header(make_project):
    folder = make_project
    make(folder)
    (folder / "nested/value.h").unlink()
    assert make(folder, check=False).returncode != 0
    next_timestamp()
    for name in ("value.cpp", "value_c.c"):
        source = folder / "nested" / name
        source.write_text(
            source.read_text().replace('#include "value.h"', "#define VALUE 24")
        )
    make(folder)
    assert output(folder) == "48"


def test_makefile_change_rebuilds_c_and_cpp(make_project):
    folder = make_project
    make(folder)
    before = {obj: obj.stat().st_mtime_ns for obj in folder.rglob("*.o")}
    next_timestamp()
    with (folder / "makefile").open("a") as stream:
        stream.write("\n# Makefile changed\n")
    make(folder)
    assert all(obj.stat().st_mtime_ns > stamp for obj, stamp in before.items())
    assert output(folder) == "42"


def test_windows_makefile_keeps_legacy_dependencies(monkeypatch):
    import brian2.devices.cpp_standalone.device as module

    generated = {}
    device = module.CPPStandaloneDevice()
    writer = SimpleNamespace(
        source_files={"main.cpp"},
        header_files={"objects.h"},
        write=lambda name, contents: generated.update({name: contents}),
    )
    with monkeypatch.context() as patch:
        patch.setattr(module, "os", SimpleNamespace(**{**vars(os), "name": "nt"}))
        device.generate_makefile(writer, "mingw32", "-O2", "", 0, False)
    text = generated["makefile"]
    assert "DEPS = make.deps" in text
    assert "-MM $(SRCS) > make.deps" in text
    assert "del *.o /s\n\tdel main.exe $(DEPS)" in text
    assert "-MMD" not in text
    assert "missing-deps" not in text
