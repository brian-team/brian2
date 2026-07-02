"""Run Brian2 runtime benchmarks in fresh subprocesses.

The subprocess boundary matters here: cppyy keeps process-local JIT state, while Cython
also has an on-disk cache. Starting a new process for each cell gives us a cleaner view
of cold-process behavior, and repeating inside that process shows warm behavior.

Results are written to:
    cppyy_optimization_work/raw_results/<label>_<timestamp>.json

Usage:
    python bench_harness.py --label baseline
    python bench_harness.py --label after_fix --targets cppyy --workloads small_lif kremer3
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPTS_DIR.parent / "raw_results"
RAW_DIR.mkdir(parents=True, exist_ok=True)

VENV_PYTHON = REPO / "venv" / "bin" / "python3"
PYTHON = str(VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable))


def child_env() -> dict[str, str]:
    """Environment shared by benchmark subprocesses."""
    env = {**os.environ}
    env.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")

    # Keep imports pinned to this checkout, even when the child process runs from
    # the scripts directory. Preserve any existing PYTHONPATH after our entries.
    py_path = [str(REPO), str(SCRIPTS_DIR)]
    if env.get("PYTHONPATH"):
        py_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_path)
    return env


def env_info() -> dict:
    """Capture enough context to understand a benchmark run later."""
    try:
        head = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(REPO), "branch", "--show-current"],
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            text=True,
        ).strip() != ""
    except Exception:
        head, branch, dirty = "unknown", "unknown", False

    versions = subprocess.check_output(
        [
            PYTHON,
            "-c",
            "import json, sys; "
            "versions = {'python': sys.version.split()[0]}; "
            "import brian2; versions['brian2'] = brian2.__version__; "
            "\ntry:\n import cppyy; versions['cppyy'] = cppyy.__version__\n"
            "except Exception as exc:\n versions['cppyy'] = 'unavailable: ' + type(exc).__name__\n"
            "try:\n import Cython; versions['cython'] = Cython.__version__\n"
            "except Exception as exc:\n versions['cython'] = 'unavailable: ' + type(exc).__name__\n"
            "print(json.dumps(versions))",
        ],
        env=child_env(),
        text=True,
    ).strip()

    return {
        "git_head": head,
        "git_branch": branch,
        "git_dirty": dirty,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_executable": PYTHON,
        "extra_cling_args": os.environ.get("EXTRA_CLING_ARGS", ""),
        **json.loads(versions),
    }


def run_one(
    target: str,
    workload: str,
    repeat: int,
    duration_ms: float,
    cython_cache: str,
) -> list[dict]:
    """Run the bench_runner subprocess and parse JSON lines."""
    cmd = [
        PYTHON,
        str(SCRIPTS_DIR / "bench_runner.py"),
        target,
        workload,
        "--repeat",
        str(repeat),
        "--duration-ms",
        str(duration_ms),
        "--cython-cache",
        cython_cache,
    ]
    proc = subprocess.run(
        cmd,
        env=child_env(),
        capture_output=True,
        text=True,
        cwd=str(SCRIPTS_DIR),
    )
    records = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if proc.returncode != 0 or not records:
        print(f"[FAILED] {target}/{workload}:", file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        print("--- stderr ---", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Brian2 Cython/cppyy benchmark workloads."
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--targets", nargs="+", default=["cython", "cppyy"])
    parser.add_argument("--workloads", nargs="+", default=["small_lif", "coba", "kremer3"])
    parser.add_argument(
        "--warm-repeat",
        type=int,
        default=3,
        help="Iterations per subprocess. Iteration 0 is cold for that process.",
    )
    parser.add_argument(
        "--cold-repeats",
        type=int,
        default=2,
        help="Number of fresh subprocesses per target/workload cell.",
    )
    parser.add_argument("--duration-ms-small", type=float, default=200.0)
    parser.add_argument("--duration-ms-coba", type=float, default=500.0)
    parser.add_argument("--duration-ms-kremer", type=float, default=200.0)
    parser.add_argument(
        "--reset-cython-cache",
        action="store_true",
        help="Delete the benchmark Cython cache before starting.",
    )
    args = parser.parse_args()

    cython_cache = str(REPO / "cythontmp_bench")
    if args.reset_cython_cache and os.path.isdir(cython_cache):
        shutil.rmtree(cython_cache)

    durations = {
        "small_lif": args.duration_ms_small,
        "coba": args.duration_ms_coba,
        "kremer3": args.duration_ms_kremer,
    }

    all_records: list[dict] = []
    meta = {
        "label": args.label,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "env": env_info(),
        "args": vars(args),
    }

    for target in args.targets:
        for workload in args.workloads:
            dur = durations.get(workload, 200.0)
            for cold_idx in range(args.cold_repeats):
                t0 = time.perf_counter()
                recs = run_one(target, workload, args.warm_repeat, dur, cython_cache)
                elapsed = time.perf_counter() - t0
                for r in recs:
                    r["cold_subprocess_idx"] = cold_idx
                    r["cold"] = (r["iter"] == 0)
                    r["subprocess_wall_s"] = elapsed if r["iter"] == 0 else None
                all_records.extend(recs)
                if recs:
                    cold = recs[0]
                    warm = recs[-1] if len(recs) > 1 else None
                    msg = (
                        f"[{target:6s}] {workload:12s} subproc#{cold_idx} "
                        f"cold sim={cold.get('sim_s', float('nan')):.3f}s"
                    )
                    if warm is not None:
                        msg += f"  warm sim={warm.get('sim_s', float('nan')):.3f}s"
                    print(msg)
                else:
                    print(f"[{target}] {workload} subproc#{cold_idx}: NO RECORDS")

    out_path = RAW_DIR / f"{args.label}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out_path.write_text(json.dumps({"meta": meta, "records": all_records}, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
