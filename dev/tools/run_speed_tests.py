from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *
from brian2.tests.features.speed import *
from brian2genn.correctness_testing import GeNNConfiguration

# Full testing
# res = run_speed_tests()
# res.plot_all_tests()
# show()

# Quick testing
res = run_speed_tests(configurations=[#NumpyConfiguration,
                                      WeaveConfiguration,
                                      CPPStandaloneConfiguration,
                                      CPPStandaloneConfigurationOpenMP,
                                      GeNNConfiguration,
                                      ],
                      speed_tests=[
                                   LinearNeuronsOnly,
                                   HHNeuronsOnly,
                                   CUBAFixedConnectivity,
                                   VerySparseMediumRateSynapsesOnly,
                                   SparseMediumRateSynapsesOnly,
                                   DenseMediumRateSynapsesOnly,
                                   SparseLowRateSynapsesOnly,
                                   SparseHighRateSynapsesOnly,
                                   ],
                      #n_slice=slice(None, None, 3),
                      #n_slice=slice(None, -1),
                      #run_twice=False,
                      )
res.plot_all_tests()
show()

# Debug
# c = GeNNConfiguration()
# c.before_run()
# f = VerySparseMediumRateSynapsesOnly(1000000)
# f.run()
# c.after_run()
