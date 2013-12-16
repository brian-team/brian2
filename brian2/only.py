'''
A dummy package to allow wildcard import from brian2 without also importing
the pylab (numpy + matplotlib) namespace.

Usage: ``from brian2.only import *``

'''

# To minimize the problems with imports, import the packages in a sensible
# order

# The units and utils package does not depend on any other Brian package and
# should be imported first
from brian2.units import *
from brian2.units.stdunits import *
from brian2.utils import *
from brian2.core.tracking import *
from brian2.core.names import *
from brian2.core.spikesource import *

# The following packages only depend on something in the above set
from brian2.core.functions import *
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
from brian2.core.namespace import *
from brian2.groups import *
from brian2.synapses import *
from brian2.monitors import *
from brian2.devices import set_device, get_device, insert_device_code

# preferences
from brian2.core.core_preferences import *
brian_prefs.load_preferences()
brian_prefs.do_validation()

brian_prefs._backup()

def restore_initial_state():
    '''
    Restores internal Brian variables to the state they are in when Brian is imported

    Resets ``defaultclock.dt = 0.1*ms``, ``defaultclock.t = 0*ms``, `clear` all
    objects and `BrianGlobalPreferences._restore` preferences.
    '''
    if hasattr(defaultclock, '_dt'):
        del defaultclock._dt
    defaultclock._force_reinit()
    clear(erase=True)
    brian_prefs._restore()

# make the test suite available via brian2.test()
from brian2.tests import run as test
