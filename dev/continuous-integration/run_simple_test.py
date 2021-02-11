# Run a simple test that uses the main simulation elements and force code
# generation to use Cython
from brian2 import prefs
from brian2.tests.test_synapses import test_transmission_simple

prefs.codegen.target = 'cython'

test_transmission_simple()
