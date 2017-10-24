from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *
from brian2.tests.features.speed import *
#from brian2genn.correctness_testing import GeNNConfiguration
import os, pickle

use_cached_results = True

if use_cached_results and os.path.exists('cached_speed_test_results.pkl'):
    with open('cached_speed_test_results.pkl', 'rb') as f:
        res = pickle.load(f)
else:
    # Full testing
    # res = run_speed_tests()

    # Quick testing
    res = run_speed_tests(configurations=[NumpyConfiguration,
                                          WeaveConfiguration,
                                          CythonConfiguration,
                                          #LocalConfiguration,
                                          CPPStandaloneConfiguration,
                                          #CPPStandaloneConfigurationOpenMP,
    #                                      GeNNConfiguration,
                                          ],
                          speed_tests=[
                                       LinearNeuronsOnly,
                                       HHNeuronsOnly,
                                       CUBAFixedConnectivity,
                                       COBAHHFixedConnectivity,
                                       VerySparseMediumRateSynapsesOnly,
                                       SparseMediumRateSynapsesOnly,
                                       DenseMediumRateSynapsesOnly,
                                       SparseLowRateSynapsesOnly,
                                       SparseHighRateSynapsesOnly,
                                       STDP,
                                       ],
                          #n_slice=slice(None, None, 3),
                          # n_slice=slice(None, -3),
                          run_twice=False,
                          maximum_run_time=1*second,
                          )

    with open('cached_speed_test_results.pkl', 'wb') as f:
        pickle.dump(res, f, -1)

res.plot_all_tests()
# res.plot_all_tests(profiling_minimum=0.15)
res.plot_all_tests(relative=True)
show()
