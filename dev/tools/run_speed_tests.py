from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *
from brian2.tests.features.speed import LinearNeuronsOnly
#from brian2genn.correctness_testing import GeNNConfiguration

# Full testing
res = run_speed_tests()
res.plot_all_tests()
show()

# Quick testing
# res = run_speed_tests(configurations=[#NumpyConfiguration,
#                                       WeaveConfiguration,
#                                       CPPStandaloneConfiguration,
#                                       GeNNConfiguration,
#                                       ],
#                       speed_tests=[LinearNeuronsOnly],
#                       n_slice=slice(None, None, 3),
#                       run_twice=False,
#                       )
# res.plot_all_tests()
# show()

# Debug
# c = DefaultConfiguration()
# c.before_run()
# f = LinearNeuronsOnly(10)
# f.run()
# c.after_run()
# c.get_last_run_time()
# print device._last_run_time
