import pytest

from brian2 import *
from brian2.devices.device import reinit_and_delete


@pytest.mark.standalone_compatible
def test_cuba():
    taum = 20 * ms
    taue = 5 * ms
    taui = 10 * ms
    Vt = -50 * mV
    Vr = -60 * mV
    El = -49 * mV

    eqs = """
    dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
    dge/dt = -ge/taue : volt
    dgi/dt = -gi/taui : volt
    """

    P = NeuronGroup(4000, eqs, threshold="v>Vt", reset="v = Vr", refractory=5 * ms)
    P.v = "Vr + rand() * (Vt - Vr)"
    P.ge = 0 * mV
    P.gi = 0 * mV

    we = (60 * 0.27 / 10) * mV  # excitatory synaptic weight (voltage)
    wi = (-20 * 4.5 / 10) * mV  # inhibitory synaptic weight
    Ce = Synapses(P, P, on_pre="ge += we")
    Ci = Synapses(P, P, on_pre="gi += wi")
    Ce.connect("i<3200", p=0.02)
    Ci.connect("i>=3200", p=0.02)

    s_mon = SpikeMonitor(P)

    run(10 * ms)

    assert len(Ce) > 0
    assert len(Ci) > 0


if __name__ == "__main__":
    test_cuba()
