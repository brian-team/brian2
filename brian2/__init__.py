'''
Brian 2.0
'''

__docformat__ = "restructuredtext en"

__version__ = '2.0dev'
__release_date__ = 'notyet'

from .base import *
from .clocks import *
from .codegen import *
from .equations import *
from .groups import *
from .network import *
from .preferences import *
from .units import *
from .units.stdunits import *
from .utils import *

brian_prefs._backup()

def restore_initial_state():
    '''
    Restores internal Brian variables to the state they are in when Brian is imported
    
    Resets ``defaultclock.dt = 0.1*ms``, ``defaultclock.t = 0*ms``, `clear` all
    objects and `BrianGlobalPreferences._restore` preferences.
    '''
    if hasattr(defaultclock, '_dt'):
        del defaultclock._dt
    defaultclock.t = 0*ms
    clear(erase=True)
    brian_prefs._restore()
    