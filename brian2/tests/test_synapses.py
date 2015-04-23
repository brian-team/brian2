import uuid
import tempfile

from nose import with_setup, SkipTest
from nose.plugins.attrib import attr
from numpy.testing.utils import assert_equal, assert_allclose, assert_raises

from brian2 import *
from brian2.codegen.translation import make_statements
from brian2.core.variables import variables_by_owner, ArrayVariable
from brian2.utils.logger import catch_logs
from brian2.utils.stringtools import get_identifiers
from brian2.devices.device import restore_device
from brian2.codegen.permutation_analysis import check_for_order_independence, OrderDependenceError


def _compare(synapses, expected):
    conn_matrix = np.zeros((len(synapses.source), len(synapses.target)))
    for _i, _j in zip(synapses.i[:], synapses.j[:]):
        conn_matrix[_i, _j] += 1

    assert_equal(conn_matrix, expected)
    # also compare the correct numbers of incoming and outgoing synapses
    incoming = conn_matrix.sum(axis=0)
    outgoing = conn_matrix.sum(axis=1)
    assert all(synapses.N_outgoing[:] == outgoing[synapses.i[:]]), 'N_outgoing returned an incorrect value'
    assert all(synapses.N_incoming[:] == incoming[synapses.j[:]]), 'N_incoming returned an incorrect value'


@attr('codegen-independent')
def test_creation():
    '''
    A basic test that creating a Synapses object works.
    '''
    G = NeuronGroup(42, 'v: 1')
    S = Synapses(G, G, 'w:1', pre='v+=w')
    # We store weakref proxys, so we can't directly compare the objects
    assert S.source.name == S.target.name == G.name
    assert len(S) == 0
    S = Synapses(G, model='w:1', pre='v+=w')
    assert S.source.name == S.target.name == G.name


@attr('codegen-independent')
def test_name_clashes():
    # Using identical names for synaptic and pre- or post-synaptic variables
    # is confusing and should be forbidden
    G1 = NeuronGroup(1, 'a : 1')
    G2 = NeuronGroup(1, 'b : 1')
    assert_raises(ValueError, lambda: Synapses(G1, G2, 'a : 1'))
    assert_raises(ValueError, lambda: Synapses(G1, G2, 'b : 1'))

    # this should all be ok
    Synapses(G1, G2, 'c : 1')
    Synapses(G1, G2, 'a_syn : 1')
    Synapses(G1, G2, 'b_syn : 1')


def test_incoming_outgoing():
    '''
    Test the count of outgoing/incoming synapses per neuron.
    (It will be also automatically tested for all connection patterns that
    use the above _compare function for testing)
    '''
    G1 = NeuronGroup(5, 'v: 1')
    G2 = NeuronGroup(5, 'v: 1')
    S = Synapses(G1, G2, 'w:1', pre='v+=w')
    S.connect([0, 0, 0, 1, 1, 2],
              [0, 1, 2, 1, 2, 3])
    # First source neuron has 3 outgoing synapses, the second 2, the third 1
    assert all(S.N_outgoing['i==0'] == 3)
    assert all(S.N_outgoing['i==1'] == 2)
    assert all(S.N_outgoing['i==2'] == 1)
    assert all(S.N_outgoing['i>2'] == 0)
    # First target neuron receives 1 input, the second+third each 2, the fourth receives 1
    assert all(S.N_incoming['j==0'] == 1)
    assert all(S.N_incoming['j==1'] == 2)
    assert all(S.N_incoming['j==2'] == 2)
    assert all(S.N_incoming['j==3'] == 1)
    assert all(S.N_incoming['j>3'] == 0)


def test_connection_arrays():
    '''
    Test connecting synapses with explictly given arrays
    '''
    G = NeuronGroup(42, 'v : 1')
    G2 = NeuronGroup(17, 'v : 1')

    # one-to-one
    expected = np.eye(len(G2))
    S = Synapses(G2)
    S.connect(np.arange(len(G2)), np.arange(len(G2)))
    _compare(S, expected)

    # full
    expected = np.ones((len(G), len(G2)))
    S = Synapses(G, G2)
    X, Y = np.meshgrid(np.arange(len(G)), np.arange(len(G2)))
    S.connect(X.flatten(), Y.flatten())
    _compare(S, expected)

    # Multiple synapses
    expected = np.zeros((len(G), len(G2)))
    expected[3, 3] = 2
    S = Synapses(G, G2)
    S.connect([3, 3], [3, 3])
    _compare(S, expected)

    # Incorrect usage
    S = Synapses(G, G2)
    assert_raises(TypeError, lambda: S.connect([1.1, 2.2], [1.1, 2.2]))
    assert_raises(TypeError, lambda: S.connect([1, 2], 'string'))
    assert_raises(TypeError, lambda: S.connect([1, 2], [1, 2], n='i'))
    assert_raises(TypeError, lambda: S.connect([1, 2]))
    assert_raises(ValueError, lambda: S.connect([1, 2, 3], [1, 2]))
    assert_raises(ValueError, lambda: S.connect(np.ones((3, 3), dtype=np.int32),
                                                np.ones((3, 1), dtype=np.int32)))
    assert_raises(ValueError, lambda: S.connect('i==j',
                                                post=np.arange(10)))
    assert_raises(TypeError, lambda: S.connect('i==j',
                                               n=object()))
    assert_raises(TypeError, lambda: S.connect('i==j',
                                               p=object()))
    assert_raises(TypeError, lambda: S.connect(object()))


@attr('cpp_standalone', 'standalone-only')
@with_setup(teardown=restore_device)
def test_connection_array_standalone():
    previous_device = get_device()
    set_device('cpp_standalone')
    # use a clock with 1s timesteps to avoid rounding issues
    G1 = SpikeGeneratorGroup(4, np.array([0, 1, 2, 3]),
                             [0, 1, 2, 3]*second, dt=1*second)
    G2 = NeuronGroup(8, 'v:1')
    S = Synapses(G1, G2, '', pre='v+=1', dt=1*second)
    S.connect([0, 1, 2, 3], [0, 2, 4, 6])
    mon = StateMonitor(G2, 'v', record=True, name='mon', dt=1*second)
    net = Network(G1, G2, S, mon)
    net.run(5*second)
    tempdir = tempfile.mkdtemp()
    device.build(directory=tempdir, compile=True, run=True, with_output=False)
    expected = np.array([[1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0]], dtype=np.float64)
    assert_equal(mon.v, expected)
    set_device(previous_device)


def test_connection_string_deterministic_basic():
    '''
    Test connecting synapses with a deterministic string expression.
    '''
    G = NeuronGroup(17, 'v : 1')
    G.v = 'i'
    G2 = NeuronGroup(4, 'v : 1')
    G2.v = '17 + i'

    # Full connection
    expected = np.ones((len(G), len(G2)))

    S = Synapses(G, G2, 'w:1', 'v+=w')
    S.connect('True')
    _compare(S, expected)


@attr('long')
def test_connection_string_deterministic():
    '''
    Test connecting synapses with a deterministic string expression.
    '''
    G = NeuronGroup(17, 'v : 1')
    G.v = 'i'
    G2 = NeuronGroup(4, 'v : 1')
    G2.v = '17 + i'

    # Full connection
    expected = np.ones((len(G), len(G2)))

    S = Synapses(G, G2, 'w:1', 'v+=w')
    S.connect(True)
    _compare(S, expected)

    S = Synapses(G, G2, 'w:1', 'v+=w')
    S.connect('True')
    _compare(S, expected)

    S = Synapses(G, G2, 'w:1', 'v+=w', connect=True)
    _compare(S, expected)

    S = Synapses(G, G2, 'w:1', 'v+=w', connect='True')
    _compare(S, expected)

    # Full connection without self-connections
    expected = np.ones((len(G), len(G))) - np.eye(len(G))

    S = Synapses(G, G, 'w:1', 'v+=w')
    S.connect('i != j')
    _compare(S, expected)

    S = Synapses(G, G, 'w:1', 'v+=w')
    S.connect('v_pre != v_post')
    _compare(S, expected)

    S = Synapses(G, G, 'w:1', 'v+=w', connect='i != j')
    _compare(S, expected)

    # One-to-one connectivity
    expected = np.eye(len(G))

    S = Synapses(G, G, 'w:1', 'v+=w')
    S.connect('i == j')
    _compare(S, expected)

    S = Synapses(G, G, 'w:1', 'v+=w')
    S.connect('v_pre == v_post')
    _compare(S, expected)

    S = Synapses(G, G, 'w:1', 'v+=w', connect='i == j')
    _compare(S, expected)

    # Everything except for the upper [2, 2] quadrant
    number = 2
    expected = np.ones((len(G), len(G)))
    expected[:number, :number] = 0
    S = Synapses(G, G, 'w:1', 'v+=w')
    S.connect('(i >= number) or (j >= number)')
    _compare(S, expected)

    S = Synapses(G, G, 'w:1', 'v+=w')
    S.connect('(i >= explicit_number) or (j >= explicit_number)',
              namespace={'explicit_number': number})
    _compare(S, expected)


def test_connection_random_basic():
    G = NeuronGroup(4, 'v: 1')
    G2 = NeuronGroup(7, 'v: 1')

    S = Synapses(G, G2, 'w:1', 'v+=w')
    S.connect(True, p=0.0)
    assert len(S) == 0
    S.connect(True, p=1.0)
    _compare(S, np.ones((len(G), len(G2))))


@attr('long')
def test_connection_random():
    '''
    Test random connections.
    '''
    G = NeuronGroup(4, 'v: 1')
    G2 = NeuronGroup(7, 'v: 1')
    # We can only test probabilities 0 and 1 for strict correctness
    S = Synapses(G, G2, 'w:1', 'v+=w')
    S.connect('rand() < 0.')
    assert len(S) == 0
    S.connect('rand() < 1.', p=1.0)
    _compare(S, np.ones((len(G), len(G2))))

    S = Synapses(G, G2, 'w:1', 'v+=w')
    S.connect(0, 0, p=0.)
    expected = np.zeros((len(G), len(G2)))
    _compare(S, expected)

    S = Synapses(G, G2, 'w:1', 'v+=w')
    S.connect(0, 0, p=1.)
    expected = np.zeros((len(G), len(G2)))
    expected[0, 0] = 1
    _compare(S, expected)

    S = Synapses(G, G2, 'w:1', 'v+=w')
    S.connect([0, 1], [0, 2], p=1.)
    expected = np.zeros((len(G), len(G2)))
    expected[0, 0] = 1
    expected[1, 2] = 1
    _compare(S, expected)

    S = Synapses(G, G2, 'w:1', 'v+=w')
    S.connect('rand() < 1.', p=1.0)
    _compare(S, np.ones((len(G), len(G2))))

    # Just make sure using values between 0 and 1 work in principle
    S = Synapses(G, G2, 'w:1', 'v+=w')
    S.connect(True, p=0.3)
    S = Synapses(G, G2, 'w:1', 'v+=w')
    S.connect('rand() < 0.3')

    S = Synapses(G, G, 'w:1', 'v+=w')
    S.connect('i!=j', p=0.0)
    assert len(S) == 0
    S.connect('i!=j', p=1.0)
    expected = np.ones((len(G), len(G))) - np.eye(len(G))
    _compare(S, expected)

    S = Synapses(G, G, 'w:1', 'v+=w')
    S.connect('i!=j', p=0.3)

    S = Synapses(G, G, 'w:1', 'v+=w')
    S.connect(0, 0, p=0.3)

    S = Synapses(G, G, 'w:1', 'v+=w')
    S.connect([0, 1], [0, 2], p=0.3)


def test_connection_multiple_synapses():
    '''
    Test multiple synapses per connection.
    '''
    G = NeuronGroup(42, 'v: 1')
    G2 = NeuronGroup(17, 'v: 1')

    S = Synapses(G, G2, 'w:1', 'v+=w')
    S.connect(True, n=0)
    assert len(S) == 0
    S.connect(True, n=2)
    _compare(S, 2*np.ones((len(G), len(G2))))

    S = Synapses(G, G2, 'w:1', 'v+=w')
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
        assert_allclose(S.w[:], expected,
                        err_msg='Assigning %r gave incorrect result' % assignment)
        S.w = 0*volt
        S.w[:] = assignment
        assert_allclose(S.w[:], expected,
                        err_msg='Assigning %r gave incorrect result' % assignment)

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
        assert_allclose(S.w[:], expected,
                        err_msg='Assigning %r gave incorrect result' % assignment)
        S.w = 0*volt
        S.w_[:] = assignment
        assert_allclose(S.w[:], expected,
                        err_msg='Assigning %r gave incorrect result' % assignment)


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
    assert len(S.w[0:, 0:]) == len(S.w[0:, 0:, 0:]) == len(G1)*len(G2)*2
    assert len(S.w[0::2, 0:]) == 3*len(G2)*2
    assert len(S.w[0, :]) == len(S.w[0, :, :]) == len(G2)*2
    assert len(S.w[0:2, :]) == len(S.w[0:2, :, :]) == 2*len(G2)*2
    assert len(S.w[:2, :]) == len(S.w[:2, :, :]) == 2*len(G2)*2
    assert len(S.w[0:4:2, :]) == len(S.w[0:4:2, :, :]) == 2*len(G2)*2
    assert len(S.w[:4:2, :]) == len(S.w[:4:2, :, :]) == 2*len(G2)*2
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

    # 1d indexing is directly indexing synapses!
    assert len(S.w[:]) == len(S.w[0:])
    assert len(S.w[[0, 1]]) == len(S.w[3:5]) == 2
    assert len(S.w[:]) == len(S.w[np.arange(len(G1)*len(G2)*2)])

    #Array-indexing (not yet supported for synapse index)
    assert_equal(S.w[:, 0:3], S.w[:, [0, 1, 2]])
    assert_equal(S.w[:, 0:3], S.w[np.arange(len(G1)), [0, 1, 2]])

    #string-based indexing
    assert_equal(S.w[0:3, :], S.w['i<3'])
    assert_equal(S.w[:, 0:3], S.w['j<3'])
    # TODO: k is not working yet
    # assert_equal(S.w[:, :, 0], S.w['k==0'])
    assert_equal(S.w[0:3, :], S.w['v_pre < 3*mV'])
    assert_equal(S.w[:, 0:3], S.w['v_post < 13*mV'])

    #invalid indices
    assert_raises(IndexError, lambda: S.w.__getitem__((1, 2, 3, 4)))
    assert_raises(IndexError, lambda: S.w.__getitem__(object()))


def test_indices():
    G = NeuronGroup(10, 'v : 1')
    S = Synapses(G, G, '', connect=True)
    G.v = 'i'

    assert_equal(S.indices[:], np.arange(10*10))
    assert len(S.indices[5, :]) == 10
    assert_equal(S.indices['v_pre >=5'], S.indices[5:, :])
    assert_equal(S.indices['j >=5'], S.indices[:, 5:])


def test_subexpression_references():
    '''
    Assure that subexpressions in targeted groups are handled correctly.
    '''
    G = NeuronGroup(10, '''v : 1
                           v2 = 2*v : 1''')
    G.v = np.arange(10)
    S = Synapses(G, G, '''w : 1
                          u = v2_post + 1 : 1
                          x = v2_pre + 1 : 1''')
    S.connect('i==(10-1-j)')
    assert_equal(S.u[:], np.arange(10)[::-1]*2+1)
    assert_equal(S.x[:], np.arange(10)*2+1)


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
    assert_equal(S.delay[:], 5*ms)
    S.connect('i==j')
    S.delay = 10*ms
    assert_equal(S.delay[:], 10*ms)
    # S.delay = '3*ms'
    # assert_equal(S.delay[:], 3*ms)

    # Invalid arguments
    assert_raises(DimensionMismatchError, lambda: Synapses(G, G, 'w:1',
                                                           pre='v+=w',
                                                           delay=5*mV))
    assert_raises(TypeError, lambda: Synapses(G, G, 'w:1', pre='v+=w',
                                              delay=object()))
    assert_raises(ValueError, lambda: Synapses(G, G, 'w:1', delay=5*ms))
    assert_raises(ValueError, lambda: Synapses(G, G, 'w:1', pre='v+=w',
                                               delay={'post': 5*ms}))

@attr('codegen-independent')
def test_pre_before_post():
    # The pre pathway should be executed before the post pathway
    G = NeuronGroup(1, '''x : 1
                          y : 1''', threshold='True')
    S = Synapses(G, G, '', pre='x=1; y=1', post='x=2', connect=True)
    run(defaultclock.dt)
    # Both pathways should have been executed, but post should have overriden
    # the x value (because it was executed later)
    assert G.x == 2
    assert G.y == 1

@attr('long')
def test_transmission():
    default_dt = defaultclock.dt
    delays = [[0, 0] * ms, [1, 1] * ms, [1, 2] * ms]
    for delay in delays:
        # Make sure that the Synapses class actually propagates spikes :)
        source = NeuronGroup(2, '''dv/dt = rate : 1
                                   rate : Hz''', threshold='v>1', reset='v=0')
        source.rate = [51, 101] * Hz
        target = NeuronGroup(2, 'v:1', threshold='v>1', reset='v=0')

        source_mon = SpikeMonitor(source)
        target_mon = SpikeMonitor(target)

        S = Synapses(source, target, pre='v+=1.1', connect='i==j')
        S.delay = delay
        net = Network(S, source, target, source_mon, target_mon)
        net.run(100*ms+default_dt+max(delay))

        # All spikes should trigger spikes in the receiving neurons with
        # the respective delay ( + one dt)
        assert_allclose(source_mon.t[source_mon.i==0],
                        target_mon.t[target_mon.i==0] - default_dt - delay[0])
        assert_allclose(source_mon.t[source_mon.i==1],
                        target_mon.t[target_mon.i==1] - default_dt - delay[1])

#@attr('standalone-compatible')  # scalar delays not yet supported in standalone
@with_setup(teardown=restore_device)
def test_transmission_scalar_delay():
    inp = SpikeGeneratorGroup(2, [0, 1], [0, 1]*ms)
    target = NeuronGroup(2, 'v:1')
    S = Synapses(inp, target, pre='v+=1', delay=0.5*ms, connect='i==j')
    mon = StateMonitor(target, 'v', record=True)
    net = Network(inp, target, S, mon)
    net.run(2*ms)
    assert_equal(mon[0].v[mon.t<0.5*ms], 0)
    assert_equal(mon[0].v[mon.t>=0.5*ms], 1)
    assert_equal(mon[1].v[mon.t<1.5*ms], 0)
    assert_equal(mon[1].v[mon.t>=1.5*ms], 1)

#@attr('standalone-compatible')  # scalar delays not yet supported in standalone
@with_setup(teardown=restore_device)
def test_transmission_scalar_delay_different_clocks():

    inp = SpikeGeneratorGroup(2, [0, 1], [0, 1]*ms, dt=0.5*ms,
                              # give the group a unique name to always
                              # get a 'fresh' warning
                              name='sg_%d' % uuid.uuid4())
    target = NeuronGroup(2, 'v:1', dt=0.1*ms)
    S = Synapses(inp, target, pre='v+=1', delay=0.5*ms, connect='i==j')
    mon = StateMonitor(target, 'v', record=True)
    net = Network(inp, target, S, mon)

    # We should get a warning when using inconsistent dts
    with catch_logs() as l:
        net.run(2*ms)
        assert len(l) == 1, 'expected a warning, got %d' % len(l)
        assert l[0][1].endswith('synapses_dt_mismatch')

    assert_equal(mon[0].v[mon.t<0.5*ms], 0)
    assert_equal(mon[0].v[mon.t>=0.5*ms], 1)
    assert_equal(mon[1].v[mon.t<1.5*ms], 0)
    assert_equal(mon[1].v[mon.t>=1.5*ms], 1)


@attr('codegen-independent')
def test_clocks():
    '''
    Make sure that a `Synapse` object uses the correct clocks.
    '''
    source_dt = 0.05*ms
    target_dt = 0.1*ms
    synapse_dt = 0.2*ms
    source = NeuronGroup(1, 'v:1', dt=source_dt)
    target = NeuronGroup(1, 'v:1', dt=target_dt)
    synapse = Synapses(source, target, 'w:1', pre='v+=1', post='v+=1',
                       dt=synapse_dt, connect=True)

    assert synapse.pre.clock is source.clock
    assert synapse.post.clock is target.clock
    assert synapse.pre._clock.dt == source_dt
    assert synapse.post._clock.dt == target_dt
    assert synapse._clock.dt == synapse_dt


@with_setup(teardown=restore_initial_state)
def test_changed_dt_spikes_in_queue():
    defaultclock.dt = .5*ms
    G1 = NeuronGroup(1, 'v:1', threshold='v>1', reset='v=0')
    G1.v = 1.1
    G2 = NeuronGroup(10, 'v:1', threshold='v>1', reset='v=0')
    S = Synapses(G1, G2, pre='v+=1.1')
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
    assert_equal(mon.t[:], expected)


@attr('codegen-independent')
def test_no_synapses():
    # Synaptic pathway but no synapses
    G1 = NeuronGroup(1, '', threshold='True')
    G2 = NeuronGroup(1, 'v:1')
    S = Synapses(G1, G2, pre='v+=1', name='synapses_'+str(uuid.uuid4()).replace('-', '_'))
    net = Network(G1, G2, S)
    with catch_logs() as l:
        net.run(defaultclock.dt)
        assert len(l) == 1, 'expected 1 warning, got %d' % len(l)
        assert l[0][1].endswith('.no_synapses')

#@attr('standalone-compatible')  # synaptic indexing is not yet possible in standalone
@with_setup(teardown=restore_device)
def test_summed_variable():
    source = NeuronGroup(2, 'v : 1', threshold='v>1', reset='v=0')
    source.v = 1.1  # will spike immediately
    target = NeuronGroup(2, 'v : 1')
    S = Synapses(source, target, '''w : 1
                                    x : 1
                                    v_post = x : 1 (summed)''', pre='x+=w')
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


def test_scalar_parameter_access():
    G = NeuronGroup(10, '''v : 1
                           scalar : Hz (shared)''')
    S = Synapses(G, G, '''w : 1
                          s : Hz (shared)
                          number : 1 (shared)''',
                 pre = 'v+=w*number', connect=True)

    # Try setting a scalar variable
    S.s = 100*Hz
    assert_equal(S.s[:], 100*Hz)
    S.s[:] = 200*Hz
    assert_equal(S.s[:], 200*Hz)
    S.s = 's - 50*Hz + number*Hz'
    assert_equal(S.s[:], 150*Hz)
    S.s[:] = '50*Hz'
    assert_equal(S.s[:], 50*Hz)

    # Set a postsynaptic scalar variable
    S.scalar_post = 100*Hz
    assert_equal(G.scalar[:], 100*Hz)
    S.scalar_post[:] = 100*Hz
    assert_equal(G.scalar[:], 100*Hz)

    # Check the second method of accessing that works
    assert_equal(np.asanyarray(S.s), 50*Hz)

    # Check error messages
    assert_raises(IndexError, lambda: S.s[0])
    assert_raises(IndexError, lambda: S.s[1])
    assert_raises(IndexError, lambda: S.s[0:1])
    assert_raises(IndexError, lambda: S.s['i>5'])

    assert_raises(ValueError, lambda: S.s.set_item(slice(None), [0, 1]*Hz))
    assert_raises(IndexError, lambda: S.s.set_item(0, 100*Hz))
    assert_raises(IndexError, lambda: S.s.set_item(1, 100*Hz))
    assert_raises(IndexError, lambda: S.s.set_item('i>5', 100*Hz))


def test_scalar_subexpression():
    G = NeuronGroup(10, '''v : 1
                           number : 1 (shared)''')
    S = Synapses(G, G, '''s : 1 (shared)
                          sub = number_post + s : 1 (shared)''',
                 pre='v+=s', connect=True)
    S.s = 100
    G.number = 50
    assert S.sub[:] == 150

    assert_raises(SyntaxError, lambda: Synapses(G, G, '''s : 1 (shared)
                                                     sub = v_post + s : 1 (shared)''',
                                                pre='v+=s', connect=True))

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_external_variables():
    # Make sure that external variables are correctly resolved
    source = SpikeGeneratorGroup(1, [0], [0]*ms)
    target = NeuronGroup(1, 'v:1')
    w_var = 1
    amplitude = 2
    syn = Synapses(source, target, 'w=w_var : 1',
                   pre='v+=amplitude*w', connect=True)
    net = Network(source, target, syn)
    net.run(defaultclock.dt)
    assert target.v[0] == 2


@attr('long', 'standalone-compatible')
@with_setup(teardown=restore_device)
def test_event_driven():
    # Fake example, where the synapse is actually not changing the state of the
    # postsynaptic neuron, the pre- and post spiketrains are regular spike
    # trains with different rates
    pre = NeuronGroup(2, '''dv/dt = rate : 1
                            rate : Hz''', threshold='v>1', reset='v=0')
    pre.rate = [1000, 1500] * Hz
    post = NeuronGroup(2, '''dv/dt = rate : 1
                             rate : Hz''', threshold='v>1', reset='v=0')
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
                  connect='i==j')
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
                  connect='i==j')
    S1.w = 0.5*gmax
    S2.w = 0.5*gmax
    net = Network(pre, post, S1, S2)
    net.run(25*ms)
    # The two formulations should yield identical results
    assert_equal(S1.w[:], S2.w[:])


@attr('codegen-independent')
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

@attr('codegen-independent')
def test_variables_by_owner():
    # Test the `variables_by_owner` convenience function
    G = NeuronGroup(10, 'v : 1')
    G2 = NeuronGroup(10, '''v : 1
                            w : 1''')
    S = Synapses(G, G2, 'x : 1')

    # Check that the variables returned as owned by the pre/post groups are the
    # variables stored in the respective groups. We only compare the `Variable`
    # objects, as the names may be different (e.g. ``v_post`` vs. ``v``)
    assert set(G.variables.values()) == set(variables_by_owner(S.variables, G).values())
    assert set(G2.variables.values()) == set(variables_by_owner(S.variables, G2).values())
    assert len(set(variables_by_owner(S.variables, S)) & set(G.variables.values())) == 0
    assert len(set(variables_by_owner(S.variables, S)) & set(G2.variables.values())) == 0
    # Just test a few examples for synaptic variables
    assert all(varname in variables_by_owner(S.variables, S)
               for varname in ['x', 'N', 'N_incoming', 'N_outgoing'])

@attr('codegen-independent')
def check_permutation_code(code):
    from collections import defaultdict
    vars = get_identifiers(code)
    indices = defaultdict(lambda: '_idx')
    for var in vars:
        if var.endswith('_syn'):
            indices[var] = '_idx'
        elif var.endswith('_pre'):
            indices[var] ='_presynaptic_idx'
        elif var.endswith('_post'):
            indices[var] = '_postsynaptic_idx'
    variables = dict()
    for var in indices:
        variables[var] = ArrayVariable(var, 1, None, 10, device)
    variables['_presynaptic_idx'] = ArrayVariable(var, 1, None, 10, device)
    variables['_postsynaptic_idx'] = ArrayVariable(var, 1, None, 10, device)
    scalar_statements, vector_statements = make_statements(code, variables, float64)
    check_for_order_independence(vector_statements, variables, indices)
    
@attr('codegen-independent')
def test_permutation_analysis():
    # Examples that should work
    good_examples = [
        'v_post += w_syn',
        'v_post *= w_syn',
        'v_post = v_post + w_syn',
        'v_post = v_post * w_syn',
        'v_post = w_syn * v_post',
        'v_post += 1',
        'v_post = 1',
        'w_syn = v_pre',
        'w_syn = a_syn',
        'w_syn += a_syn',
        'w_syn *= a_syn',
        'w_syn += 1',
        'w_syn *= 2',
        '''
        w_syn = a_syn
        a_syn += 1
        ''',
        'v_post *= 2',
        'v_post *= w_syn',
        '''
        v_pre = 0
        w_syn = v_pre
        ''',
        '''
        ge_syn += w_syn
        Apre_syn += 3
        w_syn = clip(w_syn + Apost_syn, 0, 10)
        ''',
    ]
    for example in good_examples:
        try:
            check_permutation_code(example)
        except OrderDependenceError:
            raise AssertionError(('Test unexpectedly raised an '
                                  'OrderDependenceError on these '
                                  'statements:\n') + example)

    bad_examples = [
        'v_pre = w_syn',
        'v_post = v_pre',
        '''
        a_syn = v_post
        v_post += w_syn
        '''
    ]
    for example in bad_examples:
        assert_raises(OrderDependenceError, check_permutation_code, example)


@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_vectorisation():
    source = NeuronGroup(10, 'v : 1', threshold='v>1')
    target = NeuronGroup(10, '''x : 1
                                y : 1''')
    syn = Synapses(source, target, 'w_syn : 1',
                   pre='''v_pre += w_syn
                          x_post = y_post
                       ''', connect=True)
    syn.w_syn = 1
    source.v['i<5'] = 2
    target.y = 'i'
    run(defaultclock.dt)
    assert_equal(source.v[:5], 12)
    assert_equal(source.v[5:], 0)
    assert_equal(target.x[:], target.y[:])

@attr('standalone-compatible')
@with_setup(teardown=restore_device)
def test_vectorisation_STDP_like():
    # Test the use of pre- and post-synaptic traces that are stored in the
    # pre/post group instead of in the synapses
    w_max = 10
    neurons = NeuronGroup(6, '''dv/dt = rate : 1
                                ge : 1
                                rate : Hz
                                dA/dt = -A/(1*ms) : 1''', threshold='v>1', reset='v=0')
    # Note that the synapse does not actually increase the target v, we want
    # to have simple control about when neurons spike. Also, we separate the
    # "depression" and "facilitation" completely. The example also uses
    # subgroups, which should complicate things further.
    # This test should try to capture the spirit of indexing in such a use case,
    # it simply compares the results to fixed pre-calculated values
    syn = Synapses(neurons[:3], neurons[3:], '''w_dep : 1
                                                w_fac : 1''',
                   pre='''ge_post += w_dep - w_fac
                          A_pre += 1
                          w_dep = clip(w_dep + A_post, 0, w_max)
                       ''',
                   post='''A_post += 1
                           w_fac = clip(w_fac + A_pre, 0, w_max)
                        ''', connect=True)
    neurons.rate = 1000*Hz
    neurons.v = 'abs(3-i)*0.1 + 0.7'
    run(2*ms)
    # Make sure that this test is invariant to synapse order
    indices = np.argsort(np.array(zip(syn.i[:], syn.j[:]),
                                  dtype=[('i', '<i4'), ('j', '<i4')]),
                         order=['i', 'j'])
    assert_allclose(syn.w_dep[:][indices],
                    [1.29140162, 1.16226149, 1.04603529, 1.16226149, 1.04603529,
                     0.94143176, 1.04603529, 0.94143176, 6.2472887],
                    rtol=1e-6, atol=1e-12)
    assert_allclose(syn.w_fac[:][indices],
                    [5.06030369, 5.62256002, 6.2472887, 5.62256002, 6.2472887,
                     6.941432, 6.2472887, 6.941432, 1.04603529],
                    rtol=1e-6, atol=1e-12)
    assert_allclose(neurons.A[:],
                    [1.69665715, 1.88517461, 2.09463845, 2.32737606, 2.09463845,
                     1.88517461],
                    rtol=1e-6, atol=1e-12)
    assert_allclose(neurons.ge[:],
                    [0., 0., 0., -7.31700015, -8.13000011, -4.04603529],
                    rtol=1e-6, atol=1e-12)


if __name__ == '__main__':
    test_creation()
    test_name_clashes()
    test_incoming_outgoing()
    test_connection_string_deterministic()
    test_connection_random()
    test_connection_multiple_synapses()
    test_connection_arrays()
    test_connection_array_standalone()
    restore_device()
    test_state_variable_assignment()
    test_state_variable_indexing()
    test_indices()
    test_subexpression_references()
    test_delay_specification()
    test_pre_before_post()
    test_transmission()
    test_transmission_scalar_delay()
    test_transmission_scalar_delay_different_clocks()
    test_clocks()
    test_changed_dt_spikes_in_queue()
    test_no_synapses()
    test_summed_variable()
    test_summed_variable_errors()
    test_scalar_parameter_access()
    test_scalar_subexpression()
    test_external_variables()
    test_event_driven()
    test_repr()
    test_variables_by_owner()
    test_permutation_analysis()
    test_vectorisation()
    test_vectorisation_STDP_like()
