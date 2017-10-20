'''
Check the speed of different Brian 2 configurations
'''
from brian2 import *
from brian2.tests.features import SpeedTest

__all__ = ['LinearNeuronsOnly',
           'HHNeuronsOnly',
           'CUBAFixedConnectivity',
           'COBAHHFixedConnectivity',
           'VerySparseMediumRateSynapsesOnly',
           'SparseMediumRateSynapsesOnly',
           'DenseMediumRateSynapsesOnly',
           'SparseLowRateSynapsesOnly',
           'SparseHighRateSynapsesOnly',
           'STDP'
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


class COBAHHFixedConnectivity(SpeedTest):

    category = "Full examples"
    name = "COBAHH fixed connectivity"
    tags = ["Neurons", "Synapses", "SpikeMonitor"]
    n_range = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    n_label = 'Num neurons'

    # configuration options
    duration = 1 * second

    def run(self):
        N = self.n
        area = 20000 * umetre ** 2
        Cm = (1 * ufarad * cm ** -2) * area
        gl = (5e-5 * siemens * cm ** -2) * area

        El = -60 * mV
        EK = -90 * mV
        ENa = 50 * mV
        g_na = (100 * msiemens * cm ** -2) * area
        g_kd = (30 * msiemens * cm ** -2) * area
        VT = -63 * mV
        # Time constants
        taue = 5 * ms
        taui = 10 * ms
        # Reversal potentials
        Ee = 0 * mV
        Ei = -80 * mV
        we = 6 * nS  # excitatory synaptic weight
        wi = 67 * nS  # inhibitory synaptic weight

        # The model
        eqs = Equations('''
        dv/dt = (gl*(El-v)+ge*(Ee-v)+gi*(Ei-v)-
                 g_na*(m*m*m)*h*(v-ENa)-
                 g_kd*(n*n*n*n)*(v-EK))/Cm : volt
        dm/dt = alpha_m*(1-m)-beta_m*m : 1
        dn/dt = alpha_n*(1-n)-beta_n*n : 1
        dh/dt = alpha_h*(1-h)-beta_h*h : 1
        dge/dt = -ge*(1./taue) : siemens
        dgi/dt = -gi*(1./taui) : siemens
        alpha_m = 0.32*(mV**-1)*(13*mV-v+VT)/
                 (exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
        beta_m = 0.28*(mV**-1)*(v-VT-40*mV)/
                (exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
        alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
        beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
        alpha_n = 0.032*(mV**-1)*(15*mV-v+VT)/
                 (exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
        beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
        ''')

        P = NeuronGroup(N, model=eqs, threshold='v>-20*mV',
                        refractory=3 * ms,
                        method='exponential_euler')
        P.v = 'El + (randn() * 5 - 5)*mV'
        P.ge = '(randn() * 1.5 + 4) * 10.*nS'
        P.gi = '(randn() * 12 + 20) * 10.*nS'

        Pe = P[:int(0.8*N)]
        Pi = P[int(0.8 * N):]
        Ce = Synapses(Pe, P, on_pre='ge+=we')
        Ci = Synapses(Pi, P, on_pre='gi+=wi')
        Ce.connect(p=80.0/N)
        Ci.connect(p=80.0/N)

        s_mon = SpikeMonitor(P)

        self.timed_run(self.duration)


class STDP(SpeedTest):

    category = "Full examples"
    name = "STDP with Poisson input"
    tags = ["Neurons", "Synapses", "SpikeMonitor", "PoissonGroup"]
    n_range = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    n_label = 'Num neurons'

    # configuration options
    duration = 1 * second

    def run(self):
        N = self.n
        taum = 10 * ms
        taupre = 20 * ms
        taupost = taupre
        Ee = 0 * mV
        vt = -54 * mV
        vr = -60 * mV
        El = -74 * mV
        taue = 5 * ms
        F = 15 * Hz
        gmax = .01
        dApre = .01
        dApost = -dApre * taupre / taupost * 1.05
        dApost *= gmax
        dApre *= gmax

        eqs_neurons = '''
        dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
        dge/dt = -ge / taue : 1
        '''

        input = PoissonGroup(N, rates=F)
        neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                              method='exact')
        S = Synapses(input, neurons,
                     '''w : 1
                        dApre/dt = -Apre / taupre : 1 (event-driven)
                        dApost/dt = -Apost / taupost : 1 (event-driven)''',
                     on_pre='''ge += w
                            Apre += dApre
                            w = clip(w + Apost, 0, gmax)''',
                     on_post='''Apost += dApost
                             w = clip(w + Apre, 0, gmax)''',
                     )
        S.connect()
        S.w = 'rand() * gmax'
        s_mon = SpikeMonitor(input)

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
