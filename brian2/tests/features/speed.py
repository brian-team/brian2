'''
Check the speed of different Brian 2 configurations
'''
from brian2 import *
from brian2.tests.features import SpeedTest

__all__ = ['LinearNeuronsOnly',
           'HHNeuronsOnly',
           'CUBAFixedConnectivity',
           'VerySparseMediumRateSynapsesOnly',
           'SparseMediumRateSynapsesOnly',
           'DenseMediumRateSynapsesOnly',
           'SparseLowRateSynapsesOnly',
           'SparseHighRateSynapsesOnly',
           ]


class LinearNeuronsOnly(SpeedTest):

    category = "Neurons only"
    name = "Linear 1D"
    tags = ["Neurons"]
    n_range = [10, 100, 1000, 10000, 100000, 1000000]
    n_label = 'Num neurons'

    # configuration options
    duration = 10 * second

    def run(self):
        self.tau = tau = 1 * second
        self.v_init = linspace(0.1, 1, self.n)
        G = self.G = NeuronGroup(self.n, 'dv/dt=-v/tau:1')
        self.G.v = self.v_init
        self.timed_run(self.duration)


class HHNeuronsOnly(SpeedTest):

    category = "Neurons only"
    name = "Hodgkin-Huxley"
    tags = ["Neurons"]
    n_range = [10, 100, 1000, 10000, 100000]
    n_label = 'Num neurons'

    # configuration options
    duration = 1 * second

    def run(self):
        num_neurons = self.n
        # Parameters
        area = 20000 * umetre**2
        Cm = 1 * ufarad * cm**-2 * area
        gl = 5e-5 * siemens * cm**-2 * area
        El = -65 * mV
        EK = -90 * mV
        ENa = 50 * mV
        g_na = 100 * msiemens * cm**-2 * area
        g_kd = 30 * msiemens * cm**-2 * area
        VT = -63 * mV

        # The model
        eqs = Equations('''
        dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/Cm : volt
        dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
            (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
            (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
        dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
            (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
        dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
        I : amp
        ''')
        # Threshold and refractoriness are only used for spike counting
        group = NeuronGroup(num_neurons, eqs,
                            threshold='v > -40*mV',
                            refractory='v > -40*mV')
        group.v = El
        group.I = '0.7*nA * i / num_neurons'
        self.timed_run(self.duration)


class CUBAFixedConnectivity(SpeedTest):

    category = "Full examples"
    name = "CUBA fixed connectivity"
    tags = ["Neurons", "Synapses", "SpikeMonitor"]
    n_range = [10, 100, 1000, 10000, 100000]
    n_label = 'Num neurons'

    # configuration options
    duration = 1 * second

    def run(self):
        N = self.n
        Ne = int(.8 * N)

        taum = 20 * ms
        taue = 5 * ms
        taui = 10 * ms
        Vt = -50 * mV
        Vr = -60 * mV
        El = -49 * mV

        eqs = '''
        dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
        dge/dt = -ge/taue : volt (unless refractory)
        dgi/dt = -gi/taui : volt (unless refractory)
        '''

        P = NeuronGroup(
            N, eqs, threshold='v>Vt', reset='v = Vr', refractory=5 * ms)
        P.v = 'Vr + rand() * (Vt - Vr)'
        P.ge = 0 * mV
        P.gi = 0 * mV

        we = (60 * 0.27 / 10) * mV  # excitatory synaptic weight (voltage)
        wi = (-20 * 4.5 / 10) * mV  # inhibitory synaptic weight
        Ce = Synapses(P, P, on_pre='ge += we')
        Ci = Synapses(P, P, on_pre='gi += wi')
        Ce.connect('i<Ne', p=80. / N)
        Ci.connect('i>=Ne', p=80. / N)

        s_mon = SpikeMonitor(P)

        self.timed_run(self.duration)


class SynapsesOnly(object):
    category = "Synapses only"
    tags = ["Synapses"]
    n_range = [10, 100, 1000, 10000]
    n_label = 'Num neurons'
    duration = 1 * second
    # memory usage will be approximately p**2*rate*dt*N**2*bytes_per_synapse/1024**3 GB
    # for CPU, bytes_per_synapse appears to be around 40?

    def run(self):
        N = self.n
        rate = self.rate
        M = int(rate * N * defaultclock.dt)
        if M <= 0:
            M = 1
        G = NeuronGroup(M, 'v:1', threshold='True')
        H = NeuronGroup(N, 'w:1')
        S = Synapses(G, H, on_pre='w += 1.0')
        S.connect(True, p=self.p)
        #M = SpikeMonitor(G)
        self.timed_run(self.duration,
            # report='text',
            )
        #plot(M.t/ms, M.i, ',k')


class VerySparseMediumRateSynapsesOnly(SynapsesOnly, SpeedTest):
    name = "Very sparse, medium rate (10s duration)"
    rate = 10 * Hz
    p = 0.02
    n_range = [10, 100, 1000, 10000, 100000]  # weave max CPU time should be about 20s
    duration = 10 * second


class SparseMediumRateSynapsesOnly(SynapsesOnly, SpeedTest):
    name = "Sparse, medium rate (1s duration)"
    rate = 10 * Hz
    p = 0.2
    n_range = [10, 100, 1000, 10000, 100000]  # weave max CPU time should be about 5m


class DenseMediumRateSynapsesOnly(SynapsesOnly, SpeedTest):
    name = "Dense, medium rate (1s duration)"
    rate = 10 * Hz
    p = 1.0
    n_range = [10, 100, 1000, 10000, 40000]  # weave max CPU time should be about 4m


class SparseLowRateSynapsesOnly(SynapsesOnly, SpeedTest):
    name = "Sparse, low rate (10s duration)"
    rate = 1 * Hz
    p = 0.2
    n_range = [10, 100, 1000, 10000, 100000]  # weave max CPU time should be about 20s
    duration = 10 * second


class SparseHighRateSynapsesOnly(SynapsesOnly, SpeedTest):
    name = "Sparse, high rate (1s duration)"
    rate = 100 * Hz
    p = 0.2
    n_range = [10, 100, 1000, 10000]  # weave max CPU time should be about 5m


if __name__ == '__main__':
    #prefs.codegen.target = 'numpy'
    VerySparseMediumRateSynapsesOnly(100000).run()
    show()
