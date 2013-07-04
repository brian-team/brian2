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


if __name__ == '__main__':
    test_creation()
    test_connection_string_deterministic()
