#!/usr/bin/env python
"""
The most simple case how to use standalone mode.
"""

from brian2 import *

set_device("cpp_standalone")  # ← only difference to "normal" simulation

tau = 10 * ms
eqs = """
dv/dt = (1-v)/tau : 1
"""
G = NeuronGroup(10, eqs, method="exact")
G.v = "rand()"
G.run_at("v += 5", times=[5, 50]*ms)
mon = StateMonitor(G, "v", record=True)
run(100 * ms)

plt.plot(mon.t / ms, mon.v.T)
plt.gca().set(xlabel="t (ms)", ylabel="v")
plt.show()
