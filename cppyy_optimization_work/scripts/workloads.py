"""Benchmark workloads used by the cppyy optimization scripts.

Each run_* function builds a self-contained Brian2 network and returns timing phases:

    build_s  - Python-side network construction
    setup_s  - net.run(0*ms), usually where code generation/compilation happens
    sim_s    - timed simulation run
    total_s  - build + setup + simulation

The runner configures the Brian2 target. Workloads only define the model and run it.
"""

from __future__ import annotations

import time
from typing import Any


def run_small_lif(duration_ms: float = 200.0) -> dict[str, float]:
    """Small LIF network with sparse recurrent synapses."""
    from brian2 import ms, mV, NeuronGroup, Synapses, Network, defaultclock, seed

    seed(123)
    defaultclock.dt = 0.1 * ms

    t0 = time.perf_counter()
    eqs = """
    dv/dt = (ge + gi + (-60*mV - v)) / (20*ms) : volt
    dge/dt = -ge/(5*ms) : volt
    dgi/dt = -gi/(10*ms) : volt
    """
    N = 2000
    G = NeuronGroup(
        N,
        eqs,
        threshold="v > -50*mV",
        reset="v = -60*mV",
        method="euler",
        name="Glif",
    )
    G.v = -60 * mV

    S = Synapses(G, G, "w : volt", on_pre="ge += w", name="Slif")
    S.connect(p=0.10)
    S.w = 0.4 * mV

    net = Network(G, S)
    t_build = time.perf_counter()
    net.run(0 * ms)  # forces compilation of before_run blocks
    t_setup = time.perf_counter()
    net.run(duration_ms * ms)
    t_sim = time.perf_counter()

    return {
        "build_s": t_build - t0,
        "setup_s": t_setup - t_build,
        "sim_s": t_sim - t_setup,
        "total_s": t_sim - t0,
    }


def run_coba(duration_ms: float = 500.0) -> dict[str, float]:
    """Classic COBA-style excitatory/inhibitory network."""
    from brian2 import ms, mV, NeuronGroup, Synapses, Network, defaultclock, seed

    seed(456)
    defaultclock.dt = 0.1 * ms

    t0 = time.perf_counter()
    taum = 20 * ms
    taue = 5 * ms
    taui = 10 * ms
    Vt = -50 * mV
    Vr = -60 * mV
    El = -49 * mV

    eqs = """
    dv/dt = (ge*(0*mV - v) + gi*(-80*mV - v) + (El - v)) / taum : volt (unless refractory)
    dge/dt = -ge/taue : 1
    dgi/dt = -gi/taui : 1
    """

    Ne = 3200
    Ni = 800
    G = NeuronGroup(
        Ne + Ni,
        eqs,
        threshold="v > Vt",
        reset="v = Vr",
        refractory=5 * ms,
        method="euler",
        name="Gcoba",
    )
    G.v = 'Vr + rand() * (Vt - Vr)'

    Ge = G[:Ne]
    Gi = G[Ne:]

    Ce = Synapses(Ge, G, on_pre="ge += 1.62*0.01", name="Ce")
    Ci = Synapses(Gi, G, on_pre="gi += 9.0*0.01", name="Ci")
    Ce.connect(p=0.02)
    Ci.connect(p=0.02)

    net = Network(G, Ce, Ci)
    t_build = time.perf_counter()
    net.run(0 * ms)
    t_setup = time.perf_counter()
    net.run(duration_ms * ms)
    t_sim = time.perf_counter()

    return {
        "build_s": t_build - t0,
        "setup_s": t_setup - t_build,
        "sim_s": t_sim - t_setup,
        "total_s": t_sim - t0,
    }


def run_kremer(
    duration_ms: float = 200.0,
    barrelarraysize: int = 3,
) -> dict[str, float]:
    """Kremer et al. 2011 barrel cortex example.

    barrelarraysize=3 is large enough to exercise synapses without making local
    iteration painful. The full example uses barrelarraysize=5.
    """
    from brian2 import (
        ms,
        mV,
        second,
        Hz,
        NeuronGroup,
        Synapses,
        Network,
        defaultclock,
        seed,
    )

    seed(789)
    defaultclock.dt = 0.1 * ms

    t0 = time.perf_counter()

    # Same per-barrel sizes as the Brian2 example; the benchmark scales the number
    # of barrels down to keep turnaround reasonable.
    M4, M23exc, M23inh = 22, 25, 12
    N4, N23exc, N23inh = M4 ** 2, M23exc ** 2, M23inh ** 2
    Nbarrels = barrelarraysize ** 2

    stim_change_time = 5 * ms
    Fmax = .5 / stim_change_time
    taum, taue, taui = 10 * ms, 2 * ms, 25 * ms
    El = -70 * mV
    Vt, vt_inc, tauvt = -55 * mV, 2 * mV, 50 * ms
    taup, taud = 5 * ms, 25 * ms
    Ap, Ad = .05, -.04
    EPSP, IPSP = 1 * mV, -1 * mV
    EPSC = EPSP * (taue / taum) ** (taum / (taue - taum))
    IPSC = IPSP * (taui / taum) ** (taum / (taui - taum))
    Ap, Ad = Ap * EPSC, Ad * EPSC

    eqs_layer4 = '''
    rate = int(is_active)*clip(cos(direction - selectivity), 0, inf)*Fmax: Hz
    is_active = abs((barrel_x + 0.5 - bar_x) * cos(direction) + (barrel_y + 0.5 - bar_y) * sin(direction)) < 0.5: boolean
    barrel_x : integer
    barrel_y : integer
    selectivity : 1
    bar_x = cos(direction)*(t - stim_start_time)/(5*ms) + stim_start_x : 1 (shared)
    bar_y = sin(direction)*(t - stim_start_time)/(5*ms) + stim_start_y : 1 (shared)
    direction : 1 (shared)
    stim_start_time : second (shared)
    stim_start_x : 1 (shared)
    stim_start_y : 1 (shared)
    '''
    layer4 = NeuronGroup(
        N4 * Nbarrels,
        eqs_layer4,
        threshold="rand() < rate*dt",
        method="euler",
        name="layer4",
    )
    layer4.barrel_x = '(i // N4) % barrelarraysize + 0.5'
    layer4.barrel_y = 'i // (barrelarraysize*N4) + 0.5'
    layer4.selectivity = '(i%N4)/(1.0*N4)*2*pi'

    stimradius = (11 + 1) * .5
    runner_code = '''
    direction = rand()*2*pi
    stim_start_x = barrelarraysize / 2.0 - cos(direction)*stimradius
    stim_start_y = barrelarraysize / 2.0 - sin(direction)*stimradius
    stim_start_time = t
    '''
    layer4.run_regularly(runner_code, dt=60 * ms, when='start')

    eqs_layer23 = '''
    dv/dt=(ge+gi+El-v)/taum : volt
    dge/dt=-ge/taue : volt
    dgi/dt=-gi/taui : volt
    dvt/dt=(Vt-vt)/tauvt : volt
    barrel_idx : integer
    x : 1
    y : 1
    '''
    layer23 = NeuronGroup(
        Nbarrels * (N23exc + N23inh),
        eqs_layer23,
        threshold="v>vt",
        reset="v = El; vt += vt_inc",
        refractory=2 * ms,
        method="euler",
        name="layer23",
    )
    layer23.v = El
    layer23.vt = Vt

    layer23exc = layer23[:Nbarrels * N23exc]
    layer23inh = layer23[Nbarrels * N23exc:]

    layer23exc.x = '(i % (barrelarraysize*M23exc)) * (1.0/M23exc)'
    layer23exc.y = '(i // (barrelarraysize*M23exc)) * (1.0/M23exc)'
    layer23exc.barrel_idx = 'floor(x) + floor(y)*barrelarraysize'

    layer23inh.x = 'i % (barrelarraysize*M23inh) * (1.0/M23inh)'
    layer23inh.y = 'i // (barrelarraysize*M23inh) * (1.0/M23inh)'
    layer23inh.barrel_idx = 'floor(x) + floor(y)*barrelarraysize'

    feedforward = Synapses(
        layer4,
        layer23exc,
        model='''w:volt
                 dA_source/dt = -A_source/taup : volt (event-driven)
                 dA_target/dt = -A_target/taud : volt (event-driven)''',
        on_pre='''ge+=w
                  A_source += Ap
                  w = clip(w+A_target, 0*volt, EPSC)''',
        on_post='''A_target += Ad
                   w = clip(w+A_source, 0*volt, EPSC)''',
        name="feedforward",
    )
    feedforward.connect(
        "(barrel_x_pre + barrelarraysize*barrel_y_pre) == barrel_idx_post",
        p=0.5,
    )
    feedforward.w = EPSC * .5

    recurrent_exc = Synapses(
        layer23exc,
        layer23,
        model="w:volt",
        on_pre="ge+=w",
        name="recurrent_exc",
    )
    recurrent_exc.connect(
        p=".15*exp(-.5*(((x_pre-x_post)/.4)**2+((y_pre-y_post)/.4)**2))"
    )
    recurrent_exc.w['j<Nbarrels*N23exc'] = EPSC * .3
    recurrent_exc.w['j>=Nbarrels*N23exc'] = EPSC

    recurrent_inh = Synapses(
        layer23inh,
        layer23exc,
        on_pre="gi+=IPSC",
        name="recurrent_inh",
    )
    recurrent_inh.connect(
        p="exp(-.5*(((x_pre-x_post)/.2)**2+((y_pre-y_post)/.2)**2))"
    )

    net = Network(layer4, layer23, feedforward, recurrent_exc, recurrent_inh)
    t_build = time.perf_counter()
    net.run(0 * ms)
    t_setup = time.perf_counter()
    net.run(duration_ms * ms)
    t_sim = time.perf_counter()

    return {
        "build_s": t_build - t0,
        "setup_s": t_setup - t_build,
        "sim_s": t_sim - t_setup,
        "total_s": t_sim - t0,
        "n_feedforward": int(len(feedforward)),
        "n_rec_exc": int(len(recurrent_exc)),
        "n_rec_inh": int(len(recurrent_inh)),
    }


WORKLOADS: dict[str, Any] = {
    "small_lif": run_small_lif,
    "coba": run_coba,
    "kremer3": run_kremer,
}
