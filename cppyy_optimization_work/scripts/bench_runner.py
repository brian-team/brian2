"""Run one benchmark workload in a single Python process.

Usage:
    python bench_runner.py <target> <workload> [--repeat N] [--duration-ms MS]

Configures the Brian2 target, runs the workload, prints one JSON dict per line:
    {"target": "...", "workload": "...", "iter": k, "build_s": ..., "setup_s": ..., "sim_s": ..., ...}

For each iteration we call start_scope() so Brian2 forgets the previous network.
Iteration 0 includes cold-process JIT/import state; later iterations show warm behavior
inside the same process. Use bench_harness.py when comparing targets seriously.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def configure_target(target: str, cython_cache_dir: str) -> None:
    """Set Brian2 prefs for the chosen runtime target."""
    from brian2 import prefs

    # Keep the benchmark cache inside the checkout. This makes cleanup explicit and
    # avoids surprising writes into a user's normal Brian/Cython cache.
    os.makedirs(cython_cache_dir, exist_ok=True)
    prefs.codegen.runtime.cython.cache_dir = cython_cache_dir
    prefs.codegen.target = target
    prefs.logging.console_log_level = "ERROR"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["cython", "cppyy", "numpy"])
    parser.add_argument("workload")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--duration-ms", type=float, default=None)
    parser.add_argument("--cython-cache",
                        default=os.path.abspath("cythontmp_bench"))
    args = parser.parse_args()

    configure_target(args.target, args.cython_cache)

    from brian2 import start_scope
    from workloads import WORKLOADS

    if args.workload not in WORKLOADS:
        valid = ", ".join(sorted(WORKLOADS))
        raise SystemExit(f"Unknown workload '{args.workload}'. Valid workloads: {valid}")

    fn = WORKLOADS[args.workload]
    for k in range(args.repeat):
        start_scope()
        kwargs = {}
        if args.duration_ms is not None:
            kwargs["duration_ms"] = args.duration_ms
        result = fn(**kwargs)
        rec = {
            "target": args.target,
            "workload": args.workload,
            "iter": k,
            **result,
        }
        print(json.dumps(rec), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
