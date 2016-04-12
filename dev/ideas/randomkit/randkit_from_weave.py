from brian2 import *
from scipy import weave
import brian2
import os

brian2dir, _ = os.path.split(brian2.__file__)
rkdir = os.path.join(brian2dir, 'random', 'randomkit')
rkc = os.path.join(rkdir, 'rk.cpp')
randomkitc = os.path.join(rkdir, 'randomkit.c')

code = '''
rk_state **internal_state = get_rk_state();
if(*internal_state==NULL)
{
    *internal_state = new rk_state;
    rk_error errcode = rk_randomseed(*internal_state);
    std::cout << "Allocated new random state." << std::endl;
}
std::cout << rk_double(*internal_state) << std::endl;
'''

for i in range(2):
    weave.inline(code+'\n//'+str(i)+'\n', [], {}, compiler='msvc',
                 headers=['"randomkit.h"', '"rk.h"'], sources=[rkc, randomkitc],
                 libraries=['advapi32'], include_dirs=[rkdir])
