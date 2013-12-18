# Write a configuration file with the include path, necessary for weave
import os
import sys
with open('./brian_preferences', 'w') as f:
    f.write('[codegen.runtime.weave]\n')
    f.write('include_dirs = ["%s/include"]\n' % sys.prefix)

import brian2
brian2.test()

