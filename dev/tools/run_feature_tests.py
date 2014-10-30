from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *

# Full testing
#print run_feature_tests()

# Quick testing
print run_feature_tests(configurations=[DefaultConfiguration, WeaveConfiguration])
