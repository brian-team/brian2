'''
This tool can be used to run setup.py inplace on Windows.
'''

import os
os.chdir('../../')
os.system('python setup.py build_ext --inplace')
