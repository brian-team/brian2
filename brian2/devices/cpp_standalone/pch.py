"""Conservative setup for experimental, project-local precompiled headers."""

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

HEADER = """#ifndef BRIAN_STANDALONE_PCH_H
#define BRIAN_STANDALONE_PCH_H
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <climits>
#endif
"""

MANIFEST = "brian_pch_manifest.json"
ARTIFACTS = {
    "brian_pch.h",
    "brian_pch_use.h",
    "brian_standalone.pch",
    "brian_pch_use.h.gch",
    "brian_pch.d",
    MANIFEST,
}


def owned_pch_files(writer):
    """Read only the bounded file list written by this generator."""
    path = Path(writer.project_dir) / MANIFEST
    if not path.exists():
        return set()
    manifest = json.loads(path.read_text())
    files = manifest.get("files")
    if (
        manifest.get("format") != "brian-pch-v1"
        or not isinstance(files, list)
        or not all(isinstance(name, str) and name in ARTIFACTS for name in files)
    ):
        raise ValueError("Invalid Brian PCH manifest; no cache files were removed")
    return set(files)


def remove_pch_files(writer, names):
    for name in names:
        (Path(writer.project_dir) / name).unlink(missing_ok=True)
        writer.header_files.discard(name)


def prepare_pch(writer, compiler, compiler_flags, make_args):
    """Return a configuration or an explicit reason to use a normal build."""
    if os.name == "nt" or compiler not in ("gcc", "unix"):
        return None, "only POSIX GCC and Clang are supported"
    try:
        flags = shlex.split(compiler_flags)
        command = shlex.split(os.environ.get("CXX", "g++"))
    except ValueError:
        return None, "compiler flags could not be parsed safely"
    if any(
        flag.startswith(
            (
                "-include",
                "--include",
                "-imacros",
                "--imacros",
                "-Xclang",
                "-Xpreprocessor",
                "-Wp,",
                "@",
            )
        )
        for flag in flags
    ):
        return None, "forced includes or opaque preprocessor flags are incompatible"
    if any("=" in arg or arg in ("-e", "--environment-overrides") for arg in make_args):
        return None, "make variable overrides require a normal build"
    if len(command) != 1:
        return None, "compiler wrappers are not supported by this experiment"
    executable = shutil.which(command[0])
    if executable is None or any(char.isspace() for char in executable):
        return None, "a simple compiler executable path is required"
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None, "compiler identification failed"
    version = result.stdout.strip()
    if "clang" in version.lower():
        mode = "clang"
        artifact = "brian_standalone.pch"
    elif "Free Software Foundation" in version:
        mode = "gcc"
        artifact = "brian_pch_use.h.gch"
    else:
        return None, "compiler is neither identified GCC nor Clang"
    sources = [
        source
        for source in writer.source_files
        if source.startswith("code_objects/") and source.endswith(".cpp")
    ]
    if not sources:
        return None, "no generated code objects to precompile for"
    writer.write("brian_pch.h", HEADER)
    files = ["brian_pch.h", artifact, "brian_pch.d", MANIFEST]
    if mode == "gcc":
        writer.write("brian_pch_use.h", "#error Brian PCH was not used\n")
        files.append("brian_pch_use.h")
    writer.write(
        MANIFEST, json.dumps({"format": "brian-pch-v1", "files": files}) + "\n"
    )
    return dict(
        mode=mode,
        compiler=executable,
        artifact=artifact,
        files=files,
        version=version.replace("\n", "\n# "),
    ), None
