'''
This tool can be used to run setup.py inplace on Windows.
'''

import os
os.chdir('../../')
if os.path.exists('brian2/synapses/cythonspikequeue.pyd'):
    os.remove('brian2/synapses/cythonspikequeue.pyd')
if os.path.exists('brian2/synapses/cythonspikequeue.cpp'):
    os.remove('brian2/synapses/cythonspikequeue.cpp')
os.system('python setup.py build_ext --inplace')
