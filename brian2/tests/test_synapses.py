from numpy.testing.utils import assert_equal, assert_allclose, assert_raises
import numpy as np

from brian2 import *

# We can only test C++ if weave is availabe
try:
    import scipy.weave
    codeobj_classes = [NumpyCodeObject, WeaveCodeObject]
except ImportError:
    # Can't test C++
    codeobj_classes = [NumpyCodeObject]


def _compare(synapses, expected):
    conn_matrix = np.zeros((len(synapses.source), len(synapses.target)))
    for i, j in zip(synapses.i[:], synapses.j[:]):
        conn_matrix[i, j] += 1

    assert_equal(conn_matrix, expected)


def test_creation():
    '''
    A basic test that creating a Synapses object works.
    '''
    G = NeuronGroup(42, 'v: 1')
    for codeobj_class in codeobj_classes:
        S = Synapses(G, G, 'w:1', pre='v+=w', codeobj_class=codeobj_class)
        # We store weakref proxys, so we can't directly compare the objects
        assert S.source.name == S.target.name == G.name
        assert len(S) == 0
        S = Synapses(G, model='w:1', pre='v+=w', codeobj_class=codeobj_class)
        assert S.source.name == S.target.name == G.name


def test_connection_arrays():
    '''
    Test connecting synapses with explictly given arrays
    '''
    G = NeuronGroup(42, 'v : 1')
    G2 = NeuronGroup(17, 'v : 1')

    for codeobj_class in codeobj_classes:
        # one-to-one
        expected = np.eye(len(G2))
        S = Synapses(G2, codeobj_class=codeobj_class)
        S.connect(np.arange(len(G2)), np.arange(len(G2)))
        _compare(S, expected)

        # full
        expected = np.ones((len(G), len(G2)))
        S = Synapses(G, G2, codeobj_class=codeobj_class)
        X, Y = np.meshgrid(np.arange(len(G)), np.arange(len(G2)))
        S.connect(X.flatten(), Y.flatten())
        _compare(S, expected)

        # Multiple synapses
        expected = np.zeros((len(G), len(G2)))
        expected[3, 3] = 2
        S = Synapses(G, G2, codeobj_class=codeobj_class)
        S.connect([3, 3], [3, 3])
        _compare(S, expected)

        # Incorrect usage
        S = Synapses(G, G2, codeobj_class=codeobj_class)
        assert_raises(TypeError, lambda: S.connect([1.1, 2.2], [1.1, 2.2]))
        assert_raises(TypeError, lambda: S.connect([1, 2], 'string'))
        assert_raises(TypeError, lambda: S.connect([1, 2], [1, 2], n='i'))
        assert_raises(TypeError, lambda: S.connect([1, 2]))
        assert_raises(ValueError, lambda: S.connect(np.ones((3, 3), dtype=np.int32),
                                                    np.ones((3, 1), dtype=np.int32)))
        assert_raises(ValueError, lambda: S.connect('i==j',
                                                    post=np.arange(10)))
        assert_raises(TypeError, lambda: S.connect('i==j',
                                                   n=object()))
        assert_raises(TypeError, lambda: S.connect('i==j',
                                                   p=object()))
        assert_raises(TypeError, lambda: S.connect(object()))


def test_connection_string_deterministic():
    '''
    Test connecting synapses with a deterministic string expression.
    '''
    G = NeuronGroup(42, 'v : 1')
    G.v = 'i'
    G2 = NeuronGroup(17, 'v : 1')
    G2.v = '42 + i'

    for codeobj_class in codeobj_classes:
        # Full connection
        expected = np.ones((len(G), len(G2)))

        S = Synapses(G, G2, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect(True)
        _compare(S, expected)

        S = Synapses(G, G2, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect('True')
        _compare(S, expected)

        S = Synapses(G, G2, 'w:1', 'v+=w', connect=True, codeobj_class=codeobj_class)
        _compare(S, expected)

        S = Synapses(G, G2, 'w:1', 'v+=w', connect='True', codeobj_class=codeobj_class)
        _compare(S, expected)

        # Full connection without self-connections
        expected = np.ones((len(G), len(G))) - np.eye(len(G))

        S = Synapses(G, G, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect('i != j')
        _compare(S, expected)

        S = Synapses(G, G, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect('v_pre != v_post')
        _compare(S, expected)

        S = Synapses(G, G, 'w:1', 'v+=w', connect='i != j', codeobj_class=codeobj_class)
        _compare(S, expected)

        # One-to-one connectivity
        expected = np.eye(len(G))

        S = Synapses(G, G, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect('i == j')
        _compare(S, expected)

        S = Synapses(G, G, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect('v_pre == v_post')
        _compare(S, expected)

        S = Synapses(G, G, 'w:1', 'v+=w', connect='i == j', codeobj_class=codeobj_class)
        _compare(S, expected)


def test_connection_random():
    '''
    Test random connections.
    '''
    # We can only test probabilities 0 and 1 for strict correctness
    G = NeuronGroup(42, 'v: 1')
    G2 = NeuronGroup(17, 'v: 1')

    for codeobj_class in codeobj_classes:
        S = Synapses(G, G2, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect(True, p=0.0)
        assert len(S) == 0
        S.connect(True, p=1.0)
        _compare(S, np.ones((len(G), len(G2))))

        S = Synapses(G, G2, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect('rand() < 0.')
        assert len(S) == 0
        S.connect('rand() < 1.', p=1.0)
        _compare(S, np.ones((len(G), len(G2))))

        S = Synapses(G, G2, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect(0, 0, p=0.)
        expected = np.zeros((len(G), len(G2)))
        _compare(S, expected)

        S = Synapses(G, G2, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect(0, 0, p=1.)
        expected = np.zeros((len(G), len(G2)))
        expected[0, 0] = 1
        _compare(S, expected)

        S = Synapses(G, G2, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect([0, 1], [0, 2], p=1.)
        expected = np.zeros((len(G), len(G2)))
        expected[0, 0] = 1
        expected[1, 2] = 1
        _compare(S, expected)

        S = Synapses(G, G2, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect('rand() < 1.', p=1.0)
        _compare(S, np.ones((len(G), len(G2))))

        # Just make sure using values between 0 and 1 work in principle
        S = Synapses(G, G2, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect(True, p=0.3)
        S = Synapses(G, G2, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect('rand() < 0.3')

        S = Synapses(G, G, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect('i!=j', p=0.0)
        assert len(S) == 0
        S.connect('i!=j', p=1.0)
        expected = np.ones((len(G), len(G))) - np.eye(len(G))
        _compare(S, expected)

        S = Synapses(G, G, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect('i!=j', p=0.3)

        S = Synapses(G, G, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect(0, 0, p=0.3)

        S = Synapses(G, G, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect([0, 1], [0, 2], p=0.3)


def test_connection_multiple_synapses():
    '''
    Test multiple synapses per connection.
    '''
    G = NeuronGroup(42, 'v: 1')
    G2 = NeuronGroup(17, 'v: 1')

    for codeobj_class in codeobj_classes:
        S = Synapses(G, G2, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect(True, n=0)
        assert len(S) == 0
        S.connect(True, n=2)
        _compare(S, 2*np.ones((len(G), len(G2))))

        S = Synapses(G, G2, 'w:1', 'v+=w', codeobj_class=codeobj_class)
        S.connect(True, n='j')

        _compare(S, np.arange(len(G2)).reshape(1, len(G2)).repeat(len(G),
                                                                  axis=0))


def test_state_variable_assignment():
    '''
    Assign values to state variables in various ways
    '''
    G = NeuronGroup(10, 'v: volt')
    G.v = 'i*mV'
    S = Synapses(G, G, 'w:volt')
    S.connect(True)

    # with unit checking
    assignment_expected = [
        (5*mV, np.ones(100)*5*mV),
        (7*mV, np.ones(100)*7*mV),
        (S.i[:] * mV, S.i[:]*np.ones(100)*mV),
        ('5*mV', np.ones(100)*5*mV),
        ('i*mV', np.ones(100)*S.i[:]*mV),
        ('i*mV +j*mV', S.i[:]*mV + S.j[:]*mV),
        # reference to pre- and postsynaptic state variables
        ('v_pre', S.i[:]*mV),
        ('v_post', S.j[:]*mV),
        #('i*mV + j*mV + k*mV', S.i[:]*mV + S.j[:]*mV + S.k[:]*mV) #not supported yet
    ]

    for assignment, expected in assignment_expected:
        S.w = 0*volt
        S.w = assignment
        assert_equal(S.w[:], expected,
                     'Assigning %r gave incorrect result' % assignment)
        S.w = 0*volt
        S.w[:] = assignment
        assert_equal(S.w[:], expected,
                     'Assigning %r gave incorrect result' % assignment)

    # without unit checking
    assignment_expected = [
        (5, np.ones(100)*5*volt),
        (7, np.ones(100)*7*volt),
        (S.i[:], S.i[:]*np.ones(100)*volt),
        ('5', np.ones(100)*5*volt),
        ('i', np.ones(100)*S.i[:]*volt),
        ('i +j', S.i[:]*volt + S.j[:]*volt),
        #('i + j + k', S.i[:]*volt + S.j[:]*volt + S.k[:]*volt) #not supported yet
    ]

    for assignment, expected in assignment_expected:
        S.w = 0*volt
        S.w_ = assignment
        assert_equal(S.w[:], expected,
                     'Assigning %r gave incorrect result' % assignment)
        S.w = 0*volt
        S.w_[:] = assignment
        assert_equal(S.w[:], expected,
                     'Assigning %r gave incorrect result' % assignment)


def test_state_variable_indexing():
    G1 = NeuronGroup(5, 'v:volt')
    G1.v = 'i*mV'
    G2 = NeuronGroup(7, 'v:volt')
    G2.v= '10*mV + i*mV'
    S = Synapses(G1, G2, 'w:1')
    S.connect(True, n=2)
    S.w[:, :, 0] = '5*i + j'
    S.w[:, :, 1] = '35 + 5*i + j'

    #Slicing
    assert len(S.w[:]) == len(S.w[:, :]) == len(S.w[:, :, :]) == len(G1)*len(G2)*2
    assert len(S.w[0:]) == len(S.w[0:, 0:]) == len(S.w[0:, 0:, 0:]) == len(G1)*len(G2)*2
    assert len(S.w[0::2]) == len(S.w[0::2, 0:]) == 3*len(G2)*2
    assert len(S.w[0]) == len(S.w[0, :]) == len(S.w[0, :, :]) == len(G2)*2
    assert len(S.w[0:2]) == len(S.w[0:2, :]) == len(S.w[0:2, :, :]) == 2*len(G2)*2
    assert len(S.w[:2]) == len(S.w[:2, :]) == len(S.w[:2, :, :]) == 2*len(G2)*2
    assert len(S.w[0:4:2]) == len(S.w[0:4:2, :]) == len(S.w[0:4:2, :, :]) == 2*len(G2)*2
    assert len(S.w[:4:2]) == len(S.w[:4:2, :]) == len(S.w[:4:2, :, :]) == 2*len(G2)*2
    assert len(S.w[:, 0]) == len(S.w[:, 0, :]) == len(G1)*2
    assert len(S.w[:, 0:2]) == len(S.w[:, 0:2, :]) == 2*len(G1)*2
    assert len(S.w[:, :2]) == len(S.w[:, :2, :]) == 2*len(G1)*2
    assert len(S.w[:, 0:4:2]) == len(S.w[:, 0:4:2, :]) == 2*len(G1)*2
    assert len(S.w[:, :4:2]) == len(S.w[:, :4:2, :]) == 2*len(G1)*2
    assert len(S.w[:, :, 0]) == len(G1)*len(G2)
    assert len(S.w[:, :, 0:2]) == len(G1)*len(G2)*2
    assert len(S.w[:, :, :2]) == len(G1)*len(G2)*2
    assert len(S.w[:, :, 0:2:2]) == len(G1)*len(G2)
    assert len(S.w[:, :, :2:2]) == len(G1)*len(G2)

    #Array-indexing (not yet supported for synapse index)
    assert_equal(S.w[0:3], S.w[[0, 1, 2]])
    assert_equal(S.w[0:3], S.w[[0, 1, 2], np.arange(len(G2))])
    assert_equal(S.w[:, 0:3], S.w[:, [0, 1, 2]])
    assert_equal(S.w[:, 0:3], S.w[np.arange(len(G1)), [0, 1, 2]])

    #string-based indexing
    assert_equal(S.w[0:3], S.w['i<3'])
    assert_equal(S.w[:, 0:3], S.w['j<3'])
    # TODO: k is not working yet
    # assert_equal(S.w[:, :, 0], S.w['k==0'])
    assert_equal(S.w[0:3], S.w['v_pre < 3*mV'])
    assert_equal(S.w[:, 0:3], S.w['v_post < 13*mV'])

    #invalid indices
    assert_raises(IndexError, lambda: S.w.__getitem__((1, 2, 3, 4)))
    assert_raises(IndexError, lambda: S.w.__getitem__(object()))


def test_delay_specification():
    # By default delays are state variables (i.e. arrays), but if they are
    # specified in the initializer, they are scalars.
    G = NeuronGroup(10, 'v:1')

    # Array delay
    S = Synapses(G, G, 'w:1', pre='v+=w')
    S.connect('i==j')
    assert len(S.delay[:]) == len(G)
    S.delay = 'i*ms'
    assert_equal(S.delay[:], np.arange(len(G))*ms)
    S.delay = 5*ms
    assert_equal(S.delay[:], np.ones(len(G))*5*ms)

    # Scalar delay
    S = Synapses(G, G, 'w:1', pre='v+=w', delay=5*ms)
    S.connect('i==j')
    S.delay = 10*ms
    assert_equal(S.delay[:], 10*ms)
    S.delay = '3*ms'
    assert_equal(S.delay[:], 3*ms)
    # TODO: Assignment with strings or arrays is currently possible, it only
    # takes into account the first value

    # Invalid arguments
    assert_raises(DimensionMismatchError, lambda: Synapses(G, G, 'w:1',
                                                           pre='v+=w',
                                                           delay=5*mV))
    assert_raises(TypeError, lambda: Synapses(G, G, 'w:1', pre='v+=w',
                                              delay=object()))
    assert_raises(ValueError, lambda: Synapses(G, G, 'w:1', delay=5*ms))
    assert_raises(ValueError, lambda: Synapses(G, G, 'w:1', pre='v+=w',
                                               delay={'post': 5*ms}))

def test_transmission():
    delays = [[0, 0] * ms, [1, 1] * ms, [1, 2] * ms]
    for codeobj_class, delay in zip(codeobj_classes, delays):
        # Make sure that the Synapses class actually propagates spikes :)
        source = NeuronGroup(2, '''dv/dt = rate : 1
                                   rate : Hz''', threshold='v>1', reset='v=0',
                             codeobj_class=codeobj_class)
        source.rate = [51, 101] * Hz
        target = NeuronGroup(2, 'v:1', threshold='v>1', reset='v=0',
                             codeobj_class=codeobj_class)

        source_mon = SpikeMonitor(source)
        target_mon = SpikeMonitor(target)

        S = Synapses(source, target, pre='v+=1.1', connect='i==j',
                     codeobj_class=codeobj_class)
        S.delay = delay
        net = Network(S, source, target, source_mon, target_mon)
        net.run(100*ms+defaultclock.dt+max(delay))

        # All spikes should trigger spikes in the receiving neurons with
        # the respective delay ( + one dt)
        assert_allclose(source_mon.t[source_mon.i==0],
                        target_mon.t[target_mon.i==0] - defaultclock.dt - delay[0])
        assert_allclose(source_mon.t[source_mon.i==1],
                        target_mon.t[target_mon.i==1] - defaultclock.dt - delay[1])


def test_changed_dt_spikes_in_queue():
    for codeobj_class in codeobj_classes:
        defaultclock.dt = .5*ms
        G1 = NeuronGroup(1, 'v:1', threshold='v>1', reset='v=0',
                         codeobj_class=codeobj_class)
        G1.v = 1.1
        G2 = NeuronGroup(10, 'v:1', threshold='v>1', reset='v=0',
                         codeobj_class=codeobj_class)
        S = Synapses(G1, G2, pre='v+=1.1', codeobj_class=codeobj_class)
        S.connect(True)
        S.delay = 'j*ms'
        mon = SpikeMonitor(G2)
        net = Network(G1, G2, S, mon)
        net.run(5*ms)
        defaultclock.dt = 1*ms
        net.run(3*ms)
        defaultclock.dt = 0.1*ms
        net.run(2*ms)
        # Spikes should have delays of 0, 1, 2, ... ms and always
        # trigger a spike one dt later
        expected = [0.5, 1.5, 2.5, 3.5, 4.5, # dt=0.5ms
                    6, 7, 8, #dt = 1ms
                    8.1, 9.1 #dt=0.1ms
                    ] * ms
        assert_equal(mon.t, expected)


def test_summed_variable():
    for codeobj_class in codeobj_classes:
        source = NeuronGroup(2, 'v : 1', threshold='v>1', reset='v=0',
                             codeobj_class=codeobj_class)
        source.v = 1.1  # will spike immediately
        target = NeuronGroup(2, 'v : 1', codeobj_class=codeobj_class)
        S = Synapses(source, target, '''w : 1
                                        x : 1
                                        v_post = x : 1 (summed)''', pre='x+=w',
                     codeobj_class=codeobj_class)
        S.connect('i==j', n=2)
        S.w[:, :, 0] = 'i'
        S.w[:, :, 1] = 'i + 0.5'
        net = Network(source, target, S)
        net.run(1*ms)

        # v of the target should be the sum of the two weights
        assert_equal(target.v, np.array([0.5, 2.5]))


def test_summed_variable_errors():
    G = NeuronGroup(10, '''dv/dt = -v / (10*ms) : volt
                           sub = 2*v : volt
                           p : volt''')

    # Using the (summed) flag for a differential equation or a parameter
    assert_raises(ValueError, lambda: Synapses(G, G, '''dp_post/dt = -p_post / (10*ms) : volt (summed)'''))
    assert_raises(ValueError, lambda: Synapses(G, G, '''p_post : volt (summed)'''))

    # Using the (summed) flag for a variable name without _pre or _post suffix
    assert_raises(ValueError, lambda: Synapses(G, G, '''p = 3*volt : volt (summed)'''))

    # Using the name of a variable that does not exist
    assert_raises(ValueError, lambda: Synapses(G, G, '''q_post = 3*volt : volt (summed)'''))

    # Target equation is not a parameter
    assert_raises(ValueError, lambda: Synapses(G, G, '''sub_post = 3*volt : volt (summed)'''))
    assert_raises(ValueError, lambda: Synapses(G, G, '''v_post = 3*volt : volt (summed)'''))

    # Unit mismatch between synapses and target
    assert_raises(DimensionMismatchError,
                  lambda: Synapses(G, G, '''p_post = 3*second : second (summed)'''))

    # Two summed variable equations targetting the same variable
    assert_raises(ValueError,
                  lambda: Synapses(G, G, '''p_post = 3*volt : volt (summed)
                                            p_pre = 3*volt : volt (summed)'''))


def test_event_driven():
    for codeobj_class in codeobj_classes:
        # Fake example, where the synapse is actually not changing the state of the
        # postsynaptic neuron, the pre- and post spiketrains are regular spike
        # trains with different rates
        pre = NeuronGroup(2, '''dv/dt = rate : 1
                                rate : Hz''', threshold='v>1', reset='v=0',
                          codeobj_class=codeobj_class)
        pre.rate = [1000, 1500] * Hz
        post = NeuronGroup(2, '''dv/dt = rate : 1
                                 rate : Hz''', threshold='v>1', reset='v=0',
                          codeobj_class=codeobj_class)
        post.rate = [1100, 1400] * Hz
        # event-driven formulation
        taupre = 20 * ms
        taupost = taupre
        gmax = .01
        dApre = .01
        dApost = -dApre * taupre / taupost * 1.05
        dApost *= gmax
        dApre *= gmax
        # event-driven
        S1 = Synapses(pre, post,
                      '''w : 1
                         dApre/dt = -Apre/taupre : 1 (event-driven)
                         dApost/dt = -Apost/taupost : 1 (event-driven)''',
                      pre='''Apre += dApre
                             w = clip(w+Apost, 0, gmax)''',
                      post='''Apost += dApost
                              w = clip(w+Apre, 0, gmax)''',
                      connect='i==j',
                      codeobj_class=codeobj_class)
        # not event-driven
        S2 = Synapses(pre, post,
                      '''w : 1
                         Apre : 1
                         Apost : 1''',
                      pre='''Apre=Apre*exp((lastupdate-t)/taupre)+dApre
                             Apost=Apost*exp((lastupdate-t)/taupost)
                             w = clip(w+Apost, 0, gmax)''',
                      post='''Apre=Apre*exp((lastupdate-t)/taupre)
                              Apost=Apost*exp((lastupdate-t)/taupost) +dApost
                              w = clip(w+Apre, 0, gmax)''',
                      connect='i==j',
                      codeobj_class=codeobj_class)
        S1.w = 0.5*gmax
        S2.w = 0.5*gmax
        net = Network(pre, post, S1, S2)
        net.run(100*ms)
        # The two formulations should yield identical results
        assert_equal(S1.w[:], S2.w[:])


def test_repr():
    G = NeuronGroup(1, 'v: volt')
    S = Synapses(G, G,
                 '''w : 1
                    dApre/dt = -Apre/taupre : 1 (event-driven)
                    dApost/dt = -Apost/taupost : 1 (event-driven)''',
                 pre='''Apre += dApre
                        w = clip(w+Apost, 0, gmax)''',
                 post='''Apost += dApost
                         w = clip(w+Apre, 0, gmax)''')
    # Test that string/LaTeX representations do not raise errors
    for func in [str, repr, sympy.latex]:
        assert len(func(S.equations))


if __name__ == '__main__':
    test_creation()
    test_connection_string_deterministic()
    test_connection_random()
    test_connection_multiple_synapses()
    test_state_variable_assignment()
    test_state_variable_indexing()
    test_delay_specification()
    test_transmission()
    test_changed_dt_spikes_in_queue()
    test_summed_variable()
    test_summed_variable_errors()
    test_event_driven()
    test_repr()
