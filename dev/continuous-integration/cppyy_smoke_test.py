"""
CI smoke test for the cppyy runtime code-generation backend.

Default mode (Stage 2 of the cppyy-cross-platform workflow) proves, end to
end, that the cppyy backend can:

  1. register itself (cppyy importable, 'cppyy' listed as a codegen target),
  2. generate and JIT-compile C++ for a small network,
  3. run the simulation,
  4. produce numerically correct results (checked against the analytic
     solution, plus a spike-count/synapse invariant).

``--suite`` mode (Stage 3) runs the Brian2 test suite restricted to the
cppyy codegen target (no long tests, no codegen-independent tests, no GSL).

Exit code 0 = pass, 1 = fail. Prints ``CPPYY_SMOKE_*`` markers so results
can be grepped from CI logs unambiguously.
"""

import argparse
import os
import sys


def check_backend_registered():
    """The cppyy backend registers itself only if `import cppyy` works."""
    import brian2
    from brian2.codegen.targets import codegen_targets

    print("brian2 version:", brian2.__version__)
    # class_name is None for abstract/base entries in the target set
    names = sorted(
        target.class_name for target in codegen_targets if target.class_name
    )
    print("registered codegen targets:", names)
    if "cppyy" not in names:
        print("CPPYY_SMOKE_FAIL: 'cppyy' is not among the registered targets")
        return False
    return True


def run_smoke_simulation():
    import numpy as np

    from brian2 import (
        Network,
        NeuronGroup,
        SpikeMonitor,
        Synapses,
        ms,
        prefs,
    )

    prefs.codegen.target = "cppyy"

    # Part 1: exact integration of dv/dt = -v/tau, compared with the
    # analytic solution v(t) = v0 * exp(-t/tau). Exercises code generation,
    # JIT compilation, math functions (exp) and state read-back.
    group = NeuronGroup(4, "dv/dt = -v / (10*ms) : 1", method="exact")
    v0 = np.array([0.0, 0.25, 0.5, 1.0])
    group.v = v0
    net = Network(group)
    net.run(20 * ms)

    expected = v0 * np.exp(-2.0)  # t/tau = 20 ms / 10 ms
    print("simulated v:", np.asarray(group.v))
    print("expected  v:", expected)
    if not np.allclose(group.v, expected, rtol=1e-5, atol=1e-8):
        print("CPPYY_SMOKE_FAIL: state updater result != analytic solution")
        return False

    # Confirm the run actually went through the cppyy backend rather than
    # silently falling back to another target.
    codeobj = group.state_updater.codeobj
    print("state updater code object class:", type(codeobj).__name__)
    if type(codeobj).__name__ != "CppyyCodeObject":
        print("CPPYY_SMOKE_FAIL: simulation did not use the cppyy backend")
        return False

    # Part 2: thresholding, reset, synaptic propagation (spike queue) and
    # spike monitoring. dv/dt = 100/s crosses v > 1 every ~10.1 ms with the
    # default dt of 0.1 ms, so a 100 ms run yields 9 spikes (allow 8-11 to
    # stay robust against dt/timing changes).
    source = NeuronGroup(
        1, "dv/dt = 100/second : 1", threshold="v > 1", reset="v = 0"
    )
    target = NeuronGroup(1, "w : 1")
    connection = Synapses(source, target, on_pre="w += 0.1")
    connection.connect()
    spikes = SpikeMonitor(source)
    net2 = Network(source, target, connection, spikes)
    net2.run(100 * ms)

    n_spikes = int(spikes.num_spikes)
    w_final = float(target.w[0])
    print(f"spikes recorded: {n_spikes}, target w: {w_final}")
    if not 8 <= n_spikes <= 11:
        print("CPPYY_SMOKE_FAIL: unexpected spike count")
        return False
    if abs(w_final - 0.1 * n_spikes) > 1e-6:
        print("CPPYY_SMOKE_FAIL: synaptic weight does not match spike count")
        return False

    return True


def run_suite():
    """Stage 3: Brian2 test-suite subset with codegen target = cppyy."""
    import brian2

    # xdist parallelism is skipped on Windows, mirroring what
    # dev/continuous-integration/run_test_suite.py does for the main suite.
    in_parallel = [] if os.name == "nt" else ["cppyy"]
    return bool(
        brian2.test(
            ["cppyy"],
            long_tests=False,
            test_codegen_independent=False,
            test_standalone=None,
            test_in_parallel=in_parallel,
            reset_preferences=True,
            test_GSL=False,
            additional_args=["--tb=short", "-vv"],
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="store_true",
        help="run the cppyy test-suite subset (stage 3) instead of the "
        "smoke simulation (stage 2)",
    )
    args = parser.parse_args()

    import cppyy

    print("cppyy version:", cppyy.__version__)

    if not check_backend_registered():
        return 1

    ok = run_suite() if args.suite else run_smoke_simulation()
    print("CPPYY_SMOKE_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
