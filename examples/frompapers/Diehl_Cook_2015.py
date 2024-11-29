"""
Unsupervised learning using STDP
--------------------------------
Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit
recognition using spike-timing-dependent plasticity. Frontiers in
computational neuroscience, 9, 99.

This script replicates the small 2x400-model. It has no command line
parameters. Instead, you control it by changing the constants below
the imports. Run the script with MODE set to "train" which
(eventually) creates the files theta.npy and weights.npy in the
DATA_PATH directory. Rerun it with MODE set to "observe" to create the
assign.npy file in the same directory. Finally, run "test" to create a
confusion matrix in confusion.npy. The script also creates a few
auxilliary .npy files useful for analysis. The script requires the
progressbar2 library.

MNIST_PATH should point to the directory storing the unzipped *-byte
MNIST files (e.g. from https://github.com/cvdfoundation/mnist).
For reasonable accuracy, N_TRAIN should be 50,000+ and N_OBSERVE 1,000+.

Written in 2024 by Björn A. Lindqvist <bjourne@gmail.com>
"""
from brian2 import *
from collections import defaultdict
from pathlib import Path
from progressbar import progressbar
from random import randrange, seed as rseed
from struct import unpack
import numpy as np

# Switch between "train", "observe", and "test" to tune parameters,
# observe excitatory spiking, and test accuracy, respectively.
MODE = 'test'

# Number of training, observation, and testing samples
N_TRAIN = 200_000
N_OBSERVE = 2_000
N_TEST = 1_000

# Random seed value
SEED = 42

# Storage paths
MNIST_PATH = Path('../mnist')
DATA_PATH = Path('data')

# Number of weight save points
N_SAVE_POINTS = 100

# Don't change these values unless you know what you're doing.
N_INP = 784
N_NEURONS = 400
V_EXC_REST = -65 * mV
V_INH_REST = -60 * mV
INTENSITY = 2

# Weights of exc->inh and inh->exc synapses
W_EXC_INH = 10.4
W_INH_EXC = 17.0

def save_npy(arr, path):
    arr = np.array(arr)
    print('%-9s %-15s => %-30s' % ('Saving', arr.shape, path))
    np.save(path, arr)

def load_npy(path):
    arr = np.load(path)
    print('%-9s %-30s => %-15s' % ('Loading', path, arr.shape))
    return arr

def read_mnist(training):
    tag = 'train' if training else 't10k'
    images = open(MNIST_PATH / ('%s-images-idx3-ubyte' % tag), 'rb')
    images.read(4)
    n_images = unpack('>I', images.read(4))[0]
    n_rows = unpack('>I', images.read(4))[0]
    n_cols = unpack('>I', images.read(4))[0]

    labels = open(MNIST_PATH / ('%s-labels-idx1-ubyte' % tag), 'rb')
    labels.read(4)
    n_labels = unpack('>I', labels.read(4))[0]
    x = np.frombuffer(images.read(), dtype = np.uint8)
    x = x.reshape(n_images, -1) / 8.0
    y = np.frombuffer(labels.read(), dtype = np.uint8)
    return x, y

def build_network(training):
    eqs = '''
    dv/dt = (v_rest - v + i_exc + i_inh) / tau_mem  : volt (unless refractory)
    i_exc = ge * -v                         : volt
    i_inh = gi * (v_inh_base - v)           : volt
    dge/dt = -ge/(1 * ms)                   : 1
    dgi/dt = -gi/(2 * ms)                   : 1
    dtimer/dt = 0.1                         : second
    '''
    reset = 'v = %r; timer = 0 * ms' % V_EXC_REST
    if training:
        exc_eqs = eqs + '''
        dtheta/dt = -theta / (1e7 * ms)         : volt
        '''
        arr_theta = np.ones(N_NEURONS) * 20 * mV
        reset += '; theta += 0.05 * mV'
    else:
        exc_eqs = eqs + '''
        theta                                   : volt
        '''
        arr_theta = load_npy(DATA_PATH / 'theta.npy') * volt
    exc_eqs = Equations(exc_eqs,
                        tau_mem = 100 * ms,
                        v_rest = V_EXC_REST,
                        v_inh_base = -100 * mV)
    ng_exc = NeuronGroup(
        N_NEURONS, exc_eqs,
        threshold = 'v > (theta - 72 * mV) and (timer > 5 * ms)',
        refractory = 5 * ms,
        reset = reset,
        method = 'euler',
        name = 'exc')
    ng_exc.v = V_EXC_REST - 40 * mV
    ng_exc.theta = arr_theta

    inh_eqs = Equations(eqs,
                        tau_mem = 10 * ms,
                        v_rest = V_INH_REST,
                        v_inh_base = -85 * mV)
    ng_inh = NeuronGroup(N_NEURONS, inh_eqs,
                         threshold = 'v > -40 * mV',
                         refractory = 2 * ms,
                         reset = 'v = -45 * mV',
                         method = 'euler',
                         name = 'inh')
    ng_inh.v = V_INH_REST - 40 * mV

    syns_exc_inh = Synapses(ng_exc, ng_inh,
                            model = 'w : 1',
                            on_pre = 'ge_post += w')
    syns_exc_inh.connect(j = 'i')
    syns_exc_inh.w = W_EXC_INH

    syns_inh_exc = Synapses(ng_inh, ng_exc,
                            model = 'w : 1',
                            on_pre = 'gi_post += w')
    syns_inh_exc.connect(True)

    weights = (-np.identity(N_NEURONS) + 1) * W_INH_EXC
    syns_inh_exc.w = weights.reshape(-1)

    pg_inp = PoissonGroup(N_INP, 0 * Hz, name = 'inp')

    # During training, inp->exc synapse weights are plastic.
    model = 'w : 1'
    on_post = ''
    on_pre = 'ge_post += w'
    if training:
        on_pre += '; pre = 1.; w = clip(w - 0.0001 * post1, 0, 1.0)'
        on_post += 'post2bef = post2; w = clip(w + 0.01 * pre * post2bef, 0, 1.0); post1 = 1.; post2 = 1.'
        model += '''
        post2bef                        : 1
        dpre/dt   = -pre/(20 * ms)      : 1 (event-driven)
        dpost1/dt = -post1/(20 * ms)    : 1 (event-driven)
        dpost2/dt = -post2/(40 * ms)    : 1 (event-driven)
        '''
        weights = (np.random.random(N_INP * N_NEURONS) + 0.01) * 0.3
    else:
        weights = load_npy(DATA_PATH / 'weights.npy')

    syns_inp_exc = Synapses(
        pg_inp, ng_exc,
        model = model,
        on_pre = on_pre,
        on_post = on_post,
        name = 'inp_exc'
    )
    syns_inp_exc.connect(True)
    syns_inp_exc.delay = 'rand() * 10 * ms'
    syns_inp_exc.w = weights

    exc_mon = SpikeMonitor(ng_exc, name = 'sp_exc')
    net = Network([pg_inp, ng_exc, ng_inh,
                   syns_inp_exc, syns_exc_inh, syns_inh_exc,
                   exc_mon])
    # Initialize
    net.run(0 * ms)
    return net

def show_sample(net, sample, intensity):
    exc_mon = net['sp_exc']
    prev = exc_mon.count[:]
    net['inp'].rates = sample * intensity * Hz
    net.run(350 * ms)
    # Don't count spikes occuring during the 150 ms rest.
    next = exc_mon.count[:]
    net['inp'].rates = 0 * Hz
    net.run(150 * ms)
    pat = next - prev
    cnt = np.sum(pat)
    if cnt < 5:
        return show_sample(net, sample, intensity + 1)
    return pat

def predict(groups, rates):
    return np.argmax([rates[grp].mean() for grp in groups])

def test():
    conf = np.zeros((10, 10))
    assign = np.load(DATA_PATH / 'assign.npy')
    groups = [np.where(assign == i)[0] for i in range(10)]

    X, Y = read_mnist(False)
    net = build_network(False)
    for i in progressbar(range(N_TEST)):
        ix = randrange(len(X))
        exc = show_sample(net, X[ix], INTENSITY)
        guess = predict(groups, exc)
        real = Y[ix]
        conf[real, guess] += 1

    print('Accuracy: %6.3f' % (np.trace(conf) / np.sum(conf)))
    conf = conf/conf.sum(axis=1)[:,None]
    print(np.around(conf, 2))
    save_npy(conf, DATA_PATH / 'confusion.npy')

def normalize_plastic_weights(syns):
    conns = np.reshape(syns.w, (N_INP, N_NEURONS))
    col_sums = np.sum(conns, axis = 0)
    factors = 78./ col_sums
    conns *= factors
    syns.w = conns.reshape(-1)

def stats(net):
    tick = int(defaultclock.t / defaultclock.dt)
    cnt = np.sum(net['sp_exc'].count[:])

    inp_exc = net['inp_exc']
    w_mu = np.mean(inp_exc.w)
    w_std = np.std(inp_exc.w)

    exc = net['exc']
    theta = exc.theta / mV
    theta_mu = np.mean(theta)
    theta_sig = np.std(theta)
    return [tick, cnt, w_mu, w_std, theta_mu, theta_sig]

def train():
    X, Y = read_mnist(True)
    n_samples = X.shape[0]
    net = build_network(True)
    rows = [stats(net) + [-1]]
    w_hist = [np.array(net['inp_exc'].w)]

    ratio = max(N_TRAIN // N_SAVE_POINTS, 1)
    for i in progressbar(range(N_TRAIN)):
        ix = i % n_samples
        normalize_plastic_weights(net['inp_exc'])
        show_sample(net, X[ix], INTENSITY)
        rows.append(stats(net) + [Y[ix]])
        if i % ratio == 0:
            w_hist.append(np.array(net['inp_exc'].w))

    save_npy(rows, DATA_PATH / 'train_stats.npy')
    save_npy(w_hist, DATA_PATH / 'train_w_hist.npy')
    save_npy(net['inp_exc'].w, DATA_PATH / 'weights.npy')
    save_npy(net['exc'].theta, DATA_PATH / 'theta.npy')

def observe():
    X, Y = read_mnist(True)
    n_samples = X.shape[0]
    net = build_network(False)
    rows = [stats(net) + [-1]]
    responses = defaultdict(list)

    for i in progressbar(range(N_OBSERVE)):
        ix = i % n_samples
        sample = X[ix]
        cls = Y[ix]
        exc = show_sample(net, sample, INTENSITY)
        rows.append(stats(net) + [Y[ix]])
        responses[cls].append(exc)

    res = np.zeros((10, N_NEURONS))
    for cls, vals in responses.items():
        res[cls] = np.array(vals).mean(axis = 0)

    assign = np.argmax(res, axis = 0)
    save_npy(assign, DATA_PATH / 'assign.npy')
    save_npy(rows, DATA_PATH / 'observe_stats.npy')

if __name__ == '__main__':
    seed(SEED)
    rseed(SEED)
    np.random.seed(SEED)
    DATA_PATH.mkdir(parents = True, exist_ok = True)
    cmds = dict(train = train, observe = observe, test = test)
    cmds[MODE]()
