"""Profile the cppyy target on the COBA workload.

This is intentionally small and direct. Use it after making runtime changes when the
wall-clock benchmark says "something improved" but you need to see where time moved.
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys

os.environ.setdefault("EXTRA_CLING_ARGS", " -O2")
sys.path.insert(0, os.path.dirname(__file__))

from brian2 import prefs, start_scope

prefs.codegen.runtime.cython.cache_dir = os.path.abspath("cythontmp_bench")
prefs.codegen.target = "cppyy"
prefs.codegen.string_expression_target = "cppyy"
prefs.logging.console_log_level = "ERROR"

from workloads import run_coba

# Warm up imports and cppyy's process-local state before profiling the measured run.
start_scope()
run_coba(duration_ms=100)

start_scope()
pr = cProfile.Profile()
pr.enable()
result = run_coba(duration_ms=500)
pr.disable()

print("coba result:", result)
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(30)
print(s.getvalue())

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
ps.print_stats(30)
print(s.getvalue())
