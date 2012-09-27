'''
Brian 2.0
'''

__docformat__ = "restructuredtext en"

__version__ = '2.0dev'
__release_date__ = 'notyet'

# To minimize the problems with imports, import the modules used by many other
# modules first 
from .units import *
from .units.stdunits import *
from .preferences import *
from .base import *
from .clocks import *
from .equations import *
from .groups import *
from .network import *
from .codegen import *
from .utils import *

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
    