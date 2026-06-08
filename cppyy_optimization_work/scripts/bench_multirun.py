"""Measure repeated builds of the same network in one Python process.

This is useful for checking whether a process-local cppyy compile cache helps. The
network names are stable, so repeated iterations should generate identical source.
Cython already has its normal on-disk extension cache.

Output: JSON to stdout, one record per iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def configure(target: str, cython_cache: str) -> None:
    from brian2 import prefs

    os.makedirs(cython_cache, exist_ok=True)
    prefs.codegen.runtime.cython.cache_dir = cython_cache
    prefs.codegen.target = target
    prefs.codegen.string_expression_target = target
    prefs.logging.console_log_level = "ERROR"


def build_and_run(duration_ms: float = 200.0) -> tuple[float, float, float]:
    """Build and run a stable-name COBA-like network."""
    from brian2 import ms, mV, NeuronGroup, Synapses, Network, defaultclock, seed

    seed(456)
    defaultclock.dt = 0.1 * ms
    taum, taue, taui = 20 * ms, 5 * ms, 10 * ms
    Vt, Vr, El = -50 * mV, -60 * mV, -49 * mV
    eqs = """
    dv/dt = (ge*(0*mV - v) + gi*(-80*mV - v) + (El - v)) / taum : volt (unless refractory)
    dge/dt = -ge/taue : 1
    dgi/dt = -gi/taui : 1
    """
    Ne, Ni = 1600, 400

    t0 = time.perf_counter()
    G = NeuronGroup(
        Ne + Ni,
        eqs,
        threshold="v>Vt",
        reset="v=Vr",
        refractory=5 * ms,
        method="euler",
        name="Gmr",
    )
    G.v = 'Vr + rand()*(Vt-Vr)'
    Ge, Gi = G[:Ne], G[Ne:]
    Ce = Synapses(Ge, G, on_pre="ge += 1.62*0.01", name="Cemr")
    Ce.connect(p=0.02)
    Ci = Synapses(Gi, G, on_pre="gi += 9.0*0.01", name="Cimr")
    Ci.connect(p=0.02)
    net = Network(G, Ce, Ci)
    t_build = time.perf_counter()
    net.run(0 * ms)
    t_setup = time.perf_counter()
    net.run(duration_ms * ms)
    t_sim = time.perf_counter()
    return t_build - t0, t_setup - t_build, t_sim - t_setup


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("target", choices=["cython", "cppyy"])
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--duration-ms", type=float, default=200.0)
    p.add_argument("--cython-cache", default=os.path.abspath("cythontmp_bench"))
    args = p.parse_args()
    configure(args.target, args.cython_cache)

    from brian2 import start_scope
    for i in range(args.iters):
        start_scope()
        build_s, setup_s, sim_s = build_and_run(args.duration_ms)
        rec = {
            "target": args.target,
            "iter": i,
            "build_s": build_s,
            "setup_s": setup_s,
            "sim_s": sim_s,
            "total_s": build_s + setup_s + sim_s,
        }
        print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
