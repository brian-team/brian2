from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *
from brian2.tests.features.speed import LinearNeuronsOnly

# Full testing
res = run_speed_tests()
res.plot_all_tests()
show()

# Quick testing
# res = run_speed_tests(configurations=[NumpyConfiguration,
#                                       WeaveConfiguration],
#                       speed_tests=[LinearNeuronsOnly],
#                       n_slice=slice(None, None, 2),
#                       )
# res.plot_all_tests()
# show()

# Debug
# c = DefaultConfiguration()
# c.before_run()
# f = LinearNeuronsOnly(10)
# f.run()
# c.after_run()
