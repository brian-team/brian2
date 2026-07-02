# cppyy Optimization Benchmark Scripts

These scripts are small benchmark helpers for comparing Brian2's `cppyy` runtime target
against `cython` while working on PR #1769 ...

The goal is not to produce a polished benchmark suite. The goal is to answer practical
development questions:

- Is cppyy faster or slower than Cython for a given workload?
- How much time is spent in setup/compilation versus the actual simulation?
- Does a change to `CppyyCodeObject.run_block` move the profile in the expected way?
- Does a process-local cppyy compile cache help repeated runs?

## Files

| File | Purpose |
|---|---|
| `bench_harness.py` | Main entry point. Runs each target/workload in fresh subprocesses and writes JSON results. |
| `bench_runner.py` | Runs one target/workload inside one Python process. Used by the harness. |
| `bench_multirun.py` | Rebuilds the same stable-name network repeatedly in one process to test in-process cache behavior. |
| `plot_results.py` | Reads benchmark JSON output and creates matplotlib summary figures. |
| `profile_cppyy.py` | Profiles the cppyy target on the COBA workload with `cProfile`. |
| `workloads.py` | Brian2 workloads shared by the benchmark scripts. |

## Workloads

| Workload | What it is useful for |
|---|---|
| `small_lif` | Small recurrent LIF network. Good for quick checks and Python-call overhead. |
| `coba` | COBA-style excitatory/inhibitory network. Good mid-sized sanity check. |
| `kremer3` | Reduced Kremer et al. barrel cortex example. Good for a more realistic synapse-heavy case. |

`kremer3` uses `barrelarraysize=3` to keep local iteration reasonable. The full Brian2
example uses a larger setting and is much slower.

## Environment

Run these commands from the repository root:

```bash
cd "/Volumes/Mrigesh SSD/Brain_WIP_Folder/brian2"
```

The scripts prefer the repository venv at:

```text
./venv/bin/python3
```

If that does not exist, they fall back to the Python interpreter used to launch the
script.

For cppyy optimization runs, it is usually worth setting Cling optimization flags:

```bash
export EXTRA_CLING_ARGS=" -O2"
```

The leading space is intentional; this is the form cppyy/Cling commonly accepts.

## Main Benchmark

Run the default target/workload matrix:

```bash
./venv/bin/python3 cppyy_optimization_work/scripts/bench_harness.py \
  --label baseline
```

Run only cppyy on quick workloads:

```bash
./venv/bin/python3 cppyy_optimization_work/scripts/bench_harness.py \
  --label cppyy_quick \
  --targets cppyy \
  --workloads small_lif coba \
  --cold-repeats 1 \
  --warm-repeat 2
```

Run Cython and cppyy on the reduced Kremer workload:

```bash
./venv/bin/python3 cppyy_optimization_work/scripts/bench_harness.py \
  --label kremer_check \
  --targets cython cppyy \
  --workloads kremer3 \
  --duration-ms-kremer 200
```

Results are written to:

```text
cppyy_optimization_work/raw_results/<label>_<timestamp>.json
```

Each JSON file contains:

- `meta`: git/environment information and command arguments
- `records`: one timing record per target/workload/iteration

Important timing fields:

- `build_s`: Python-side network construction
- `setup_s`: `net.run(0*ms)`, usually where code generation and compilation happen
- `sim_s`: timed simulation run
- `total_s`: build + setup + simulation

## Plotting Results

After running `bench_harness.py`, generate summary figures with:

```bash
./venv/bin/python3 cppyy_optimization_work/scripts/plot_results.py
```

By default, the script reads:

```text
cppyy_optimization_work/raw_results/*.json
```

and writes figures to:

```text
cppyy_optimization_work/figures/
```

The main figures are:

- `cold_total_comparison.png`
- `warm_sim_progression.png`
- `warm_ratio_progression.png`

To plot only specific result files:

```bash
./venv/bin/python3 cppyy_optimization_work/scripts/plot_results.py \
  cppyy_optimization_work/raw_results/baseline_*.json \
  cppyy_optimization_work/raw_results/final_*.json
```

To choose which cppyy result labels appear in the progression plots:

```bash
./venv/bin/python3 cppyy_optimization_work/scripts/plot_results.py \
  --progression-labels baseline after_ABC v2_clean \
  --cython-label final \
  --cold-label cold
```

The default progression labels are chosen from common optimization runs when present:

```text
baseline, after_ABC, v2_clean
```

For the repeated same-network experiment, capture JSONL output first:

```bash
./venv/bin/python3 cppyy_optimization_work/scripts/bench_multirun.py cython \
  --iters 5 > /tmp/multirun_cython.jsonl

./venv/bin/python3 cppyy_optimization_work/scripts/bench_multirun.py cppyy \
  --iters 5 > /tmp/multirun_cppyy.jsonl
```

Then pass those files to the plotter:

```bash
./venv/bin/python3 cppyy_optimization_work/scripts/plot_results.py \
  --multirun-cython /tmp/multirun_cython.jsonl \
  --multirun-cppyy /tmp/multirun_cppyy.jsonl
```

That also writes:

- `multirun_comparison.png`

## Cold vs Warm Runs

`bench_harness.py` has two levels of repetition:

- `--cold-repeats`: how many fresh subprocesses to launch for each target/workload
- `--warm-repeat`: how many iterations to run inside each subprocess

Inside each subprocess:

- `iter == 0` is marked as `cold: true`
- later iterations are marked as `cold: false`

This distinction matters because:

- cppyy keeps JIT state inside the Python process
- Cython uses an on-disk extension cache
- warm Cython runs can be much faster than the first compile

To delete the benchmark Cython cache before a run:

```bash
./venv/bin/python3 cppyy_optimization_work/scripts/bench_harness.py \
  --label cold_cython \
  --reset-cython-cache
```

The cache lives at:

```text
cythontmp_bench/
```

## Profiling cppyy

Use this after a change improves wall-clock time and you want to see where time moved:

```bash
./venv/bin/python3 cppyy_optimization_work/scripts/profile_cppyy.py
```

The script warms up cppyy first, then profiles a COBA run and prints:

- top functions by cumulative time
- top functions by own time

The main thing to watch is whether time stays in `CppyyCodeObject.run_block`, moves into
cppyy dispatch, or moves into actual simulation work.

## In-Process Cache Experiment

`bench_multirun.py` rebuilds the same stable-name network repeatedly in one Python
process:

```bash
./venv/bin/python3 cppyy_optimization_work/scripts/bench_multirun.py cppyy \
  --iters 5 \
  --duration-ms 200
```

For comparison:

```bash
./venv/bin/python3 cppyy_optimization_work/scripts/bench_multirun.py cython \
  --iters 5 \
  --duration-ms 200
```

This script sets `prefs.codegen.string_expression_target` to the selected target on
purpose. It is meant to stress repeated generated-code behavior, not necessarily mirror
the default user-facing Brian2 configuration.

## Notes When Reading Results

Do not compare a single number too aggressively. These benchmarks are noisy because they
include Python setup, random connectivity generation, JIT/compiler state, and OS effects.

Use the scripts to compare patterns:

- before vs after a code change
- cppyy vs Cython on the same workload
- cold subprocess vs warm repeat
- setup time vs simulation time

The most useful cppyy optimization signal is usually:

```text
sim_s decreases while the model output shape and synapse counts stay comparable
```

For `kremer3`, the record includes synapse counts:

- `n_feedforward`
- `n_rec_exc`
- `n_rec_inh`

These are useful sanity checks when comparing runs.
