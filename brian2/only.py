"""
A dummy package to allow wildcard import from brian2 without also importing
the pylab (numpy + matplotlib) namespace.

Usage: ``from brian2.only import *``

"""
# To minimize the problems with imports, import the packages in a sensible
# order

# The units and utils package does not depend on any other Brian package and
# should be imported first

from brian2.units import *
from brian2.utils import *
from brian2.core.tracking import *
from brian2.core.names import *
from brian2.core.spikesource import *

# The following packages only depend on something in the above set
from brian2.core.variables import linked_var
from brian2.core.functions import *
from brian2.core.preferences import *
from brian2.core.clocks import *
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
from brian2.groups.subgroup import *
from brian2.synapses import *
from brian2.monitors import *
from brian2.importexport import *
from brian2.input import *
from brian2.spatialneuron import *
from brian2.devices import set_device, get_device, device, all_devices, seed
import brian2.devices.cpp_standalone as _cpp_standalone

# preferences
import brian2.core.core_preferences as _core_preferences

prefs.load_preferences()
prefs.do_validation()

prefs._backup()

set_device(all_devices["runtime"])


def restore_initial_state():
    """
    Restores internal Brian variables to the state they are in when Brian is imported

    Resets ``defaultclock.dt = 0.1*ms``,
    `BrianGlobalPreferences._restore` preferences, and set
    `BrianObject._scope_current_key` back to 0.
    """
    import gc

    prefs._restore()
    BrianObject._scope_current_key = 0
    defaultclock.dt = 0.1 * ms
    gc.collect()


# make the test suite available via brian2.test()
from brian2.tests import run as test

from brian2.units import __all__ as _all_units

__all__ = [
    "get_logger",
    "BrianLogger",
    "std_silent",
    "Trackable",
    "Nameable",
    "SpikeSource",
    "linked_var",
    "DEFAULT_FUNCTIONS",
    "Function",
    "implementation",
    "declare_types",
    "PreferenceError",
    "BrianPreference",
    "prefs",
    "brian_prefs",
    "Clock",
    "defaultclock",
    "Equations",
    "Expression",
    "Statements",
    "BrianObject",
    "BrianObjectException",
    "Network",
    "profiling_summary",
    "scheduling_summary",
    "MagicNetwork",
    "magic_network",
    "MagicError",
    "run",
    "stop",
    "collect",
    "store",
    "restore",
    "start_scope",
    "NetworkOperation",
    "network_operation",
    "StateUpdateMethod",
    "linear",
    "exact",
    "independent",
    "milstein",
    "heun",
    "euler",
    "rk2",
    "rk4",
    "ExplicitStateUpdater",
    "exponential_euler",
    "gsl_rk2",
    "gsl_rk4",
    "gsl_rkf45",
    "gsl_rkck",
    "gsl_rk8pd",
    "NumpyCodeObject",
    "CythonCodeObject",
    "get_local_namespace",
    "DEFAULT_FUNCTIONS",
    "DEFAULT_UNITS",
    "DEFAULT_CONSTANTS",
    "CodeRunner",
    "Group",
    "VariableOwner",
    "NeuronGroup",
    "Subgroup",
    "Synapses",
    "SpikeMonitor",
    "EventMonitor",
    "StateMonitor",
    "PopulationRateMonitor",
    "ImportExport",
    "BinomialFunction",
    "PoissonGroup",
    "PoissonInput",
    "SpikeGeneratorGroup",
    "TimedArray",
    "Morphology",
    "Soma",
    "Cylinder",
    "Section",
    "SpatialNeuron",
    "set_device",
    "get_device",
    "device",
    "all_devices",
    "seed",
    "restore_initial_state",
    "test",
]
__all__.extend(_all_units)
