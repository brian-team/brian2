from numpy.testing.utils import assert_equal
import numpy as np

from brian2 import *

# We can only test C++ if weave is availabe
try:
    import scipy.weave
    languages = [PythonLanguage(), CPPLanguage()]
except ImportError:
    # Can't test C++
    languages = [PythonLanguage()]


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
    for language in languages:
        S = Synapses(G, G, 'w:1', pre='v+=w', language=language)
        assert len(S) == 0


def test_connection_string_deterministic():
    '''
    Test connecting synapses with a deterministic string expression.
    '''
    G = NeuronGroup(42, 'v: 1')
    G2 = NeuronGroup(17, 'v: 1')

    for language in languages:
        # Full connection
        expected = np.ones((len(G), len(G2)))

        S = Synapses(G, G2, 'w:1', 'v+=w', language=language)
        S.connect(True)
        _compare(S, expected)

        S = Synapses(G, G2, 'w:1', 'v+=w', language=language)
        S.connect('True')
        _compare(S, expected)

        S = Synapses(G, G2, 'w:1', 'v+=w', connect=True, language=language)
        _compare(S, expected)

        S = Synapses(G, G2, 'w:1', 'v+=w', connect='True', language=language)
        _compare(S, expected)

        # Full connection without self-connections
        expected = np.ones((len(G), len(G))) - np.eye(len(G))

        S = Synapses(G, G, 'w:1', 'v+=w', language=language)
        S.connect('i != j')
        _compare(S, expected)

        S = Synapses(G, G, 'w:1', 'v+=w', connect='i != j', language=language)
        _compare(S, expected)

        # One-to-one connectivity
        expected = np.eye(len(G))

        S = Synapses(G, G, 'w:1', 'v+=w', language=language)
        S.connect('i == j')
        _compare(S, expected)

        S = Synapses(G, G, 'w:1', 'v+=w', connect='i == j', language=language)
        _compare(S, expected)


def test_connection_random():
    '''
    Test random connections.
    '''
    # We can only test probabilities 0 and 1 for strict correctness
    G = NeuronGroup(42, 'v: 1')
    G2 = NeuronGroup(17, 'v: 1')

    for language in languages:
        S = Synapses(G, G2, 'w:1', 'v+=w', language=language)
        S.connect(True, p=0.0)
        assert len(S) == 0
        S.connect(True, p=1.0)
        _compare(S, np.ones((len(G), len(G2))))

        S = Synapses(G, G2, 'w:1', 'v+=w', language=language)
        S.connect('rand() < 0.')
        assert len(S) == 0
        S.connect('rand() < 1.', p=1.0)
        _compare(S, np.ones((len(G), len(G2))))

        # Just make sure using values between 0 and 1 work in principle
        S = Synapses(G, G2, 'w:1', 'v+=w', language=language)
        S.connect(True, p=0.3)
        S = Synapses(G, G2, 'w:1', 'v+=w', language=language)
        S.connect('rand() < 0.3')

        S = Synapses(G, G, 'w:1', 'v+=w', language=language)
        S.connect('i!=j', p=0.0)
        assert len(S) == 0
        S.connect('i!=j', p=1.0)
        expected = np.ones((len(G), len(G))) - np.eye(len(G))
        _compare(S, expected)

        S = Synapses(G, G, 'w:1', 'v+=w', language=language)
        S.connect('i!=j', p=0.3)


def test_connection_multiple_synapses():
    '''
    Test multiple synapses per connection.
    '''
    G = NeuronGroup(42, 'v: 1')
    G2 = NeuronGroup(17, 'v: 1')

    for language in languages:
        S = Synapses(G, G2, 'w:1', 'v+=w', language=language)
        S.connect(True, n=0)
        assert len(S) == 0
        S.connect(True, n=2)
        _compare(S, 2*np.ones((len(G), len(G2))))

        S = Synapses(G, G2, 'w:1', 'v+=w', language=language)
        S.connect(True, n='j')

        _compare(S, np.arange(len(G2)).reshape(1, len(G2)).repeat(len(G),
                                                                  axis=0))


def test_state_variable_assignment():
    '''
    Assign values to state variables in various ways
    '''
    G = NeuronGroup(10, 'v: volt')
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
    G1 = NeuronGroup(5, 'v:1')
    G2 = NeuronGroup(7, 'v:1')
    S = Synapses(G1, G2, 'w:1')
    S.connect(True, n=2)
    S.w[:, :, 0] = '5*i + j'
    S.w[:, :, 1] = '35 + 5*i + j'

    #Slicing
    assert len(S.w[:]) == len(S.w[:, :]) == len(S.w[:, :, :]) == len(G1)*len(G2)*2
    assert len(S.w[0]) == len(S.w[0, :]) == len(S.w[0, :, :]) == len(G2)*2
    assert len(S.w[0:2]) == len(S.w[0:2, :]) == len(S.w[0:2, :, :]) == 2*len(G2)*2
    assert len(S.w[:, 0]) == len(S.w[:, 0, :]) == len(G1)*2
    assert len(S.w[:, 0:2]) == len(S.w[:, 0:2, :]) == 2*len(G1)*2
    assert len(S.w[:, :, 0]) == len(G1)*len(G2)
    assert len(S.w[:, :, 0:2]) == len(G1)*len(G2)*2

    #Array-indexing (not yet supported for synapse index)
    assert_equal(S.w[0:3], S.w[[0, 1, 2]])
    assert_equal(S.w[0:3], S.w[[0, 1, 2], np.arange(len(G2))])
    assert_equal(S.w[:, 0:3], S.w[:, [0, 1, 2]])
    assert_equal(S.w[:, 0:3], S.w[np.arange(len(G1)), [0, 1, 2]])

    #string-based indexing
    assert_equal(S.w[0:3], S.w['i<3'])
    assert_equal(S.w[:, 0:3], S.w['j<3'])
    assert_equal(S.w[:, :, 0], S.w['k==0'])


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


if __name__ == '__main__':
    test_creation()
    test_connection_string_deterministic()
    test_connection_random()
    test_connection_multiple_synapses()
    test_state_variable_assignment()
    test_state_variable_indexing()
    test_delay_specification()