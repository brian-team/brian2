'''
Brian 2.0
'''

__docformat__ = "restructuredtext en"

__version__ = '2.0a'
__release_date__ = '2013/10/12'

# Check basic dependencies
import sys
missing = []
try:
    import numpy
except ImportError as ex:
    sys.stderr.write('Importing numpy failed: %s\n' % ex)
    missing.append('numpy')
try:
    import scipy
except ImportError as ex:
    sys.stderr.write('Importing scipy failed: %s\n' % ex)
    missing.append('scipy')
try:
    import sympy
except ImportError as ex:
    sys.stderr.write('Importing sympy failed: %s\n' % ex)
    missing.append('sympy')
try:
    import pyparsing
except ImportError as ex:
    sys.stderr.write('Importing pyparsing failed: %s\n' % ex)
    missing.append('pyparsing')

if len(missing):
    raise ImportError('Some required dependencies are missing:\n' + ', '.join(missing))

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
from brian2.devices import set_device

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
    defaultclock.__init__()
    clear(erase=True)
    brian_prefs._restore()

# make the test suite available via brian2.test()
from brian2.tests import run as test
