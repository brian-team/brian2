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
        ["make", "-j2", *targets],
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
    if any(shutil.which(tool) is None for tool in ("make", "c++")):
        pytest.skip("Requires make and a C++ compiler")
    (tmp_path / "nested").mkdir()
    writer = CPPWriter(str(tmp_path))
    writer.write("nested/value.h", "#define VALUE 21\n")
    writer.write(
        "nested/value.cpp", '#include "value.h"\nint cpp_value() { return VALUE; }\n'
    )
    writer.write(
        "main.cpp",
        """#include <iostream>
int cpp_value();
int main() { std::cout << cpp_value() << std::endl; }
""",
    )
    CPPStandaloneDevice().generate_makefile(writer, "unix", "-O0", "", 0, False)
    return tmp_path


def test_makefile_clean_and_noop(make_project):
    folder = make_project
    make(folder, "clean")  # Cleaning a never-built project is valid.
    make(folder)
    assert output(folder) == "21"
    objects = list(folder.rglob("*.o"))
    assert len(objects) == 2
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
    assert output(folder) == "21"


def test_makefile_nested_header_change(make_project):
    folder = make_project
    make(folder)
    main_time = (folder / "main.o").stat().st_mtime_ns
    next_timestamp()
    (folder / "nested/value.h").write_text("#define VALUE 22\n")
    make(folder)
    assert output(folder) == "22"
    assert (folder / "main.o").stat().st_mtime_ns == main_time


@pytest.mark.parametrize("suffix", ["value", "all"])
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
    assert output(folder) == "23"


def test_makefile_missing_dependency_direct_target(make_project):
    folder = make_project
    make(folder)
    dep = folder / "nested/value.d"
    dep.unlink()
    make(folder, "nested/value.o")
    assert dep.exists()


def test_makefile_deleted_header(make_project):
    folder = make_project
    make(folder)
    (folder / "nested/value.h").unlink()
    assert make(folder, check=False).returncode != 0
    next_timestamp()
    source = folder / "nested/value.cpp"
    source.write_text(
        source.read_text().replace('#include "value.h"', "#define VALUE 24")
    )
    make(folder)
    assert output(folder) == "24"


def test_makefile_change_rebuilds_objects(make_project):
    folder = make_project
    make(folder)
    before = {obj: obj.stat().st_mtime_ns for obj in folder.rglob("*.o")}
    next_timestamp()
    with (folder / "makefile").open("a") as stream:
        stream.write("\n# Makefile changed\n")
    make(folder)
    assert all(obj.stat().st_mtime_ns > stamp for obj, stamp in before.items())
    assert output(folder) == "21"


def test_cpp_sources_only(tmp_path):
    from brian2.devices.cpp_standalone.device import CPPStandaloneDevice, CPPWriter

    writer = CPPWriter(str(tmp_path))
    writer.write("legacy.c", "")
    writer.write("main.cpp", "")
    assert writer.source_files == {"main.cpp"}
    device = CPPStandaloneDevice()
    device.build_on_run = False
    with pytest.raises(ValueError, match=r"only supports \.cpp source files"):
        device.build(directory=str(tmp_path), additional_source_files=["legacy.c"])


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_dependency_cleanup_uses_current_device_hook(tmp_path, monkeypatch, platform):
    import brian2.devices.cpp_standalone.device as module

    device = module.CPPStandaloneDevice()
    device.project_dir = str(tmp_path)
    device.writer = module.CPPWriter(str(tmp_path))
    device.writer.write("main.cpp", "")
    (tmp_path / "main.d").touch()
    (tmp_path / "unrelated.d").touch()
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform=platform))
    files = device.code_files_to_delete()
    assert ("main.d" in files) == (platform != "win32")
    assert "unrelated.d" not in files
    assert "make.deps" not in files
    (tmp_path / "make.deps").touch()
    assert ("make.deps" in device.code_files_to_delete()) == (platform != "win32")


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
