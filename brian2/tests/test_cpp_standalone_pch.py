"""Eligibility gates for experimental standalone precompiled headers."""

import json
import os
import time
from types import SimpleNamespace

import pytest

from brian2.devices.cpp_standalone.pch import prepare_pch

pytestmark = [
    pytest.mark.cpp_standalone,
    pytest.mark.skipif(os.name == "nt", reason="PCH is POSIX-only"),
]


class Writer:
    source_files = {"code_objects/example.cpp", "external.cpp"}
    code_object_sources = {"code_objects/example.cpp"}

    def __init__(self):
        self.written = {}

    def write(self, name, contents):
        self.written[name] = contents


@pytest.mark.parametrize(
    "flags",
    [
        "-include config.h",
        "-includeconfig.h",
        "-imacros config.h",
        "-include-pch prior.pch",
        "@flags.txt",
        "-Xclang -load",
        "-Wp,-include,config.h",
    ],
)
def test_pch_rejects_opaque_preprocessing(flags):
    writer = Writer()
    config, reason = prepare_pch(writer, "unix", flags, ["-j2"])
    assert config is None and reason
    assert not writer.written


@pytest.mark.parametrize("args", [["CXX=g++"], ["CXXFLAGS=-O0"], ["-e"]])
def test_pch_rejects_make_overrides(args):
    writer = Writer()
    config, reason = prepare_pch(writer, "unix", "-O3", args)
    assert config is None and reason
    assert not writer.written


def test_pch_rejects_unsupported_compiler():
    writer = Writer()
    config, reason = prepare_pch(writer, "msvc", "", [])
    assert config is None and reason
    assert not writer.written


def test_pch_rejects_compiler_wrapper(monkeypatch):
    monkeypatch.setenv("CXX", "ccache g++")
    config, reason = prepare_pch(Writer(), "unix", "-O3", [])
    assert config is None and "wrappers" in reason


@pytest.mark.parametrize(
    "version,mode",
    [
        ("Apple clang version 21", "clang"),
        ("g++ version 13\nFree Software Foundation", "gcc"),
    ],
)
def test_pch_detects_compiler_and_writes_headers(monkeypatch, version, mode):
    from brian2.devices.cpp_standalone import pch

    monkeypatch.setenv("CXX", "g++")
    monkeypatch.setattr(pch.shutil, "which", lambda name: "/test/g++")
    monkeypatch.setattr(
        pch.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=version)
    )
    writer = Writer()
    config, reason = prepare_pch(writer, "unix", "-O3 -DVALUE=7", [])
    assert reason is None and config["mode"] == mode
    assert "brian_pch.h" in writer.written
    assert ("brian_pch_use.h" in writer.written) == (mode == "gcc")
    assert config["objects"] == "code_objects/example.o"


def test_pch_ownership_is_project_local(tmp_path, monkeypatch):
    from brian2 import prefs
    from brian2.devices.cpp_standalone import pch
    from brian2.devices.cpp_standalone.device import CPPStandaloneDevice, CPPWriter

    monkeypatch.setenv("CXX", "g++")
    monkeypatch.setattr(pch.shutil, "which", lambda name: "/test/g++")
    old_pref = prefs.devices.cpp_standalone.use_precompiled_headers
    prefs.devices.cpp_standalone.use_precompiled_headers = True
    device = CPPStandaloneDevice()
    try:
        for name, version in (
            ("gcc", "g++ version 13\nFree Software Foundation"),
            ("clang", "Apple clang version 21"),
        ):
            project = tmp_path / name
            project.mkdir()
            writer = CPPWriter(str(project))
            writer.source_files.add("code_objects/example.cpp")
            writer.code_object_sources = {"code_objects/example.cpp"}
            if name == "clang":
                sentinel = project / "brian_pch_use.h"
                sentinel.write_text("user-owned")
            monkeypatch.setattr(
                pch.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=version)
            )
            device.generate_makefile(writer, "unix", "-O0", "", 0, False)
            device.project_dir, device.writer = str(project), writer
            assert device._pch_files == pch.owned_pch_files(writer)
        device.delete(code=True, data=False, run_args=False, directory=False)
        assert sentinel.read_text() == "user-owned"
    finally:
        prefs.devices.cpp_standalone.use_precompiled_headers = old_pref


@pytest.mark.standalone_only
@pytest.mark.parametrize("enabled", [False, True])
def test_pch_excludes_additional_code_object_sources(tmp_path, enabled):
    import numpy as np

    import brian2 as b

    source = tmp_path / "code_objects" / "external.cpp"
    source.parent.mkdir()
    source.write_text(
        "struct Clock { int ticks; };\nint clock_size() { return sizeof(Clock); }\n"
    )
    old_pref = b.prefs.devices.cpp_standalone.use_precompiled_headers
    try:
        b.device.reinit()
        b.set_device("cpp_standalone", build_on_run=False)
        b.start_scope()
        b.prefs.devices.cpp_standalone.use_precompiled_headers = enabled
        group = b.NeuronGroup(4, "v : 1")
        group.v = "2*i+7"
        b.Network(group).run(0.1 * b.ms)
        b.device.build(
            directory=str(tmp_path),
            additional_source_files=["code_objects/external.cpp"],
        )
        np.testing.assert_array_equal(group.v[:], np.arange(4) * 2 + 7)
        if enabled:
            objects = next(
                line
                for line in (tmp_path / "makefile").read_text().splitlines()
                if line.startswith("PCH_OBJS =")
            )
            assert "code_objects/external.o" not in objects
    finally:
        b.prefs.devices.cpp_standalone.use_precompiled_headers = old_pref


def test_pch_rejects_unknown_compiler(monkeypatch):
    from brian2.devices.cpp_standalone import pch

    monkeypatch.setenv("CXX", "g++")
    monkeypatch.setattr(pch.shutil, "which", lambda name: "/test/g++")
    monkeypatch.setattr(
        pch.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="Unknown tool")
    )
    writer = Writer()
    config, reason = prepare_pch(writer, "unix", "-O3", [])
    assert config is None and reason
    assert not writer.written


def test_pch_manifest_rejects_unknown_files(tmp_path):
    from brian2.devices.cpp_standalone.pch import MANIFEST, owned_pch_files

    writer = SimpleNamespace(project_dir=str(tmp_path))
    sentinel = tmp_path / "user.txt"
    sentinel.write_text("keep")
    (tmp_path / MANIFEST).write_text(
        json.dumps({"format": "brian-pch-v1", "files": ["user.txt"]})
    )
    with pytest.raises(ValueError, match="Invalid Brian PCH manifest"):
        owned_pch_files(writer)
    assert sentinel.read_text() == "keep"


def test_pch_manifest_cleanup_preserves_unowned_files(tmp_path):
    from brian2.devices.cpp_standalone.pch import (
        MANIFEST,
        owned_pch_files,
        remove_pch_files,
    )

    writer = SimpleNamespace(project_dir=str(tmp_path), header_files={"brian_pch.h"})
    (tmp_path / "user.pch").write_text("keep")
    (tmp_path / "brian_pch.h").write_text("owned")
    (tmp_path / MANIFEST).write_text(
        json.dumps({"format": "brian-pch-v1", "files": ["brian_pch.h", MANIFEST]})
    )
    remove_pch_files(writer, owned_pch_files(writer))
    assert (tmp_path / "user.pch").read_text() == "keep"
    assert not (tmp_path / "brian_pch.h").exists()
    assert not writer.header_files


@pytest.mark.standalone_only
def test_pch_toggle_existing_project(tmp_path):
    import numpy as np

    import brian2 as b

    old_pref = b.prefs.devices.cpp_standalone.use_precompiled_headers
    old_jobs = b.prefs.devices.cpp_standalone.extra_make_args_unix
    try:
        for enabled in (True, False, True):
            if tmp_path.joinpath("makefile").exists():
                # GNU make 3.81 compares whole-second source timestamps.
                time.sleep(1.1)
            b.device.reinit()
            b.set_device("cpp_standalone", build_on_run=False)
            b.start_scope()
            b.prefs.devices.cpp_standalone.use_precompiled_headers = enabled
            b.prefs.devices.cpp_standalone.extra_make_args_unix = ["-j2"]
            group = b.NeuronGroup(4, "v : 1", name="pch_toggle_neurons")
            group.v = "2*i+7"
            b.Network(group).run(0.1 * b.ms)
            b.device.build(directory=str(tmp_path), compile=True, run=True)
            np.testing.assert_array_equal(group.v[:], np.arange(4) * 2 + 7)
            artifacts = list(tmp_path.glob("*.pch")) + list(tmp_path.glob("*.gch"))
            assert bool(artifacts) == enabled
        (tmp_path / "user.pch").write_text("keep")
        b.device.delete(code=True, data=False, directory=False)
        assert (tmp_path / "user.pch").read_text() == "keep"
        assert not list(tmp_path.glob("brian_pch*"))
        assert not (tmp_path / "brian_standalone.pch").exists()
    finally:
        b.prefs.devices.cpp_standalone.use_precompiled_headers = old_pref
        b.prefs.devices.cpp_standalone.extra_make_args_unix = old_jobs


def test_code_cleanup_includes_pch_and_dependency_files(tmp_path):
    from brian2.devices.cpp_standalone.device import CPPStandaloneDevice

    device = CPPStandaloneDevice()
    device.project_dir = str(tmp_path)
    device.writer = SimpleNamespace(
        source_files={"main.cpp"}, header_files={"brian_pch.h"}
    )
    device._pch_files = {"brian_pch.h", "brian_standalone.pch", "missing.pch"}
    for filename in ("brian_pch.h", "brian_standalone.pch", "main.d", "user.pch"):
        (tmp_path / filename).write_text("test")

    files = device.code_files_to_delete()
    assert "brian_standalone.pch" in files
    assert "main.d" in files
    assert files.count("brian_pch.h") == 1
    assert "missing.pch" not in files
    assert "user.pch" not in files
