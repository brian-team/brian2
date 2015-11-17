from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *
from brian2.tests.features.speed import *
#from brian2genn.correctness_testing import GeNNConfiguration
import os, pickle

use_cached_results = True

if use_cached_results and os.path.exists('cached_speed_test_results.pkl'):
    res = pickle.load(open('cached_speed_test_results.pkl', 'rb'))
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
                                       VerySparseMediumRateSynapsesOnly,
                                       SparseMediumRateSynapsesOnly,
                                       DenseMediumRateSynapsesOnly,
                                       SparseLowRateSynapsesOnly,
                                       SparseHighRateSynapsesOnly,
                                       ],
                          #n_slice=slice(None, None, 3),
                          # n_slice=slice(None, -3),
                          run_twice=False,
                          maximum_run_time=1*second,
                          )

    pickle.dump(res, open('cached_speed_test_results.pkl', 'wb'), -1)

res.plot_all_tests()
# res.plot_all_tests(profiling_minimum=0.15)
res.plot_all_tests(relative=True)
show()
