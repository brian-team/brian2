from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *

# Full testing
#print run_feature_tests().tables_and_exceptions

# Quick testing
#print run_feature_tests(configurations=[DefaultConfiguration, WeaveConfiguration]).tables_and_exceptions

# Specific testing
from brian2.tests.features.synapses import SynapsesSTDP
print run_feature_tests(feature_tests=[SynapsesSTDP]).tables_and_exceptions
