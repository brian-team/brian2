from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *
from brian2.tests.features.speed import *
#from brian2genn.correctness_testing import GeNNConfiguration

# Full testing
# res = run_speed_tests()
# res.plot_all_tests()
# show()

# Quick testing
res = run_speed_tests(configurations=[NumpyConfiguration,
                                      #WeaveConfiguration,
                                      CythonConfiguration,
                                      #LocalConfiguration,
                                      #CPPStandaloneConfiguration,
                                      #CPPStandaloneConfigurationOpenMP,
#                                      GeNNConfiguration,
                                      ],
                      speed_tests=[
                                   #LinearNeuronsOnly,
                                   #HHNeuronsOnly,
                                   CUBAFixedConnectivity,
                                   #VerySparseMediumRateSynapsesOnly,
                                   #SparseMediumRateSynapsesOnly,
                                   #DenseMediumRateSynapsesOnly,
                                   #SparseLowRateSynapsesOnly,
                                   #SparseHighRateSynapsesOnly,
                                   ],
                      #n_slice=slice(None, None, 3),
                      n_slice=slice(None, -3),
                      run_twice=False,
                      maximum_run_time=1*second,
                      )
# res.plot_all_tests()
# res.plot_all_tests(relative=True)
res.plot_all_tests(profiling=True)
show()

# Debug
# from brian2.tests.features.base import results
# for x in results(LocalConfiguration, LinearNeuronsOnly, 10, maximum_run_time=10*second):
#     print x
