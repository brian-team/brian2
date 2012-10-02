'''
Brian 2.0
'''

__docformat__ = "restructuredtext en"

__version__ = '2.0dev'
__release_date__ = 'notyet'

# To minimize the problems with imports, import the packages in a sensible
# order

# The units and utils package does not depend on any other Brian package and
# should be imported first 
from brian2.units import *
from brian2.units.stdunits import *
from brian2.utils import *

# The following packages only depend on something in the above set
from brian2.core.preferences import *
from brian2.core.clocks import *
from brian2.core.scheduler import *
from brian2.equations import *

# The base class only depends on the above sets
from brian2.core.base import *

# The rest...
from brian2.core.network import *
from brian2.core.magic import *
from brian2.core.operations import *
from brian2.stateupdaters import *
from brian2.codegen import *
from brian2.groups import *

# TODO: this will probably be moved into the codegen module, just here as a sample preference
brian_prefs.define('weave_compiler', 'gcc',
    '''
    The compiler name to use for ``scipy.weave``.
    
    On Linux platforms, gcc is usually available. On Windows, it can be made
    available as part of the mingw and cygwin packages, and also included in
    some Python distributions such as EPD and Python(x,y).
    ''')

brian_prefs._backup()

def restore_initial_state():
    '''
    Restores internal Brian variables to the state they are in when Brian is imported
    
    Resets ``defaultclock.dt = 0.1*ms``, ``defaultclock.t = 0*ms``, `clear` all
    objects and `BrianGlobalPreferences._restore` preferences.
    '''
    if hasattr(defaultclock, '_dt'):
        del defaultclock._dt
    defaultclock.__init__()
    clear(erase=True)
    brian_prefs._restore()
    