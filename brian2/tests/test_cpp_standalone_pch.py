"""Eligibility gates for experimental standalone precompiled headers."""

import json
import os
from types import SimpleNamespace

import pytest

from brian2.devices.cpp_standalone.pch import prepare_pch

pytestmark = [
    pytest.mark.cpp_standalone,
    pytest.mark.skipif(os.name == "nt", reason="PCH is POSIX-only"),
]


class Writer:
    source_files = {"code_objects/example.cpp", "external.cpp"}

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
