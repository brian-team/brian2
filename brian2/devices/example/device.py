'''
Module implementing the "example" device. This is a device for documentation
purposes that is not useful in practice. It will write out a Python script that
should perform the same simulation as the script simulated with the example
device.
'''
import numpy as np

from brian2.core.namespace import get_local_namespace, DEFAULT_UNITS
from brian2.core.variables import Constant
from brian2.equations.equations import DIFFERENTIAL_EQUATION, SUBEXPRESSION, \
    PARAMETER
from brian2.groups.neurongroup import NeuronGroup
from brian2.monitors.spikemonitor import SpikeMonitor
from brian2.devices.device import Device, all_devices
from brian2.units import second, Unit, Quantity

from brian2.utils.stringtools import get_identifiers
from brian2.utils.logger import get_logger
__all__ = []

logger = get_logger(__name__)


class DummyCodeObject(object):
    def __init__(self, *args, **kwds):
        pass

    def __call__(self, **kwds):
        pass


class ExampleDevice(Device):
    '''
    The `Device` used for C++ standalone simulations.
    '''
    def __init__(self):
        super(ExampleDevice, self).__init__()
        self.runs = []
        self.assignments = []

    # Our device will not actually calculate/store any data, so the following
    # are just dummy implementations

    def add_array(self, var):
        pass

    def init_with_zeros(self, var, dtype):
        pass

    def fill_with_array(self, var, arr):
        pass

    def init_with_arange(self, var, start, dtype):
        pass

    def get_value(self, var, access_data=True):
        return np.zeros(var.size, dtype=var.dtype)

    def resize(self, var, new_size):
        pass

    def code_object(self, *args, **kwds):
        return DummyCodeObject(*args, **kwds)

    def network_run(self, network, duration, report=None, report_period=10*second,
                    namespace=None, profile=True, level=0):
        network._clocks = {obj.clock for obj in network.objects}
        # Get the local namespace
        if namespace is None:
            namespace = get_local_namespace(level=level+2)
        network.before_run(namespace)

        # Extract all the objects present in the network
        descriptions = []
        merged_namespace = {}
        for obj in network.objects:
            one_description, one_namespace = description(obj, namespace)
            descriptions.append((obj.name, one_description))
            for key, value in one_namespace.iteritems():
                if key in merged_namespace and value != merged_namespace[key]:
                    raise ValueError('name "%s" is used inconsistently')
                merged_namespace[key] = value

        assignments = list(self.assignments)
        self.assignments[:] = []
        self.runs.append((descriptions, duration, merged_namespace, assignments))

    def variableview_set_with_expression_conditional(self, variableview, cond, code,
                                                     run_namespace, check_units=True):
        self.assignments.append(('conditional', variableview.group.name, variableview.name, cond, code))

    def variableview_set_with_expression(self, variableview, item, code, run_namespace, check_units=True):
        self.assignments.append(('item', variableview.group.name, variableview.name, item, code))

    def variableview_set_with_index_array(self, variableview, item, value, check_units):
        self.assignments.append(('item', variableview.group.name, variableview.name, item, value))

    def build(self):
        code_lines = ['from brian2 import *', '']
        created = set()
        for descriptions, duration, namespace, assignments in self.runs:
            # Create all the necessary objects if they have not been already
            # created for earlier runs
            for name, description in descriptions:
                if name not in created and len(description):
                    code_lines.append(description)
                created.add(name)
            code_lines += ['']
            # Perform all assignments
            for assignment in assignments:
                if assignment[0] == 'conditional':
                    group_name, var_name, condition, code = assignment[1:]
                    if condition == 'True':
                        code_lines.append('{}.{} = {!r}'.format(group_name,
                                                                var_name,
                                                                code))
                    else:
                        code_lines.append('{}.{}[{!r}] = {!r}'.format(group_name,
                                                                      var_name,
                                                                      condition,
                                                                      code))
                elif assignment[0] == 'item':
                    group_name, var_name, item, code = assignment[1:]
                    code_lines.append('{}.{}[{!r}] = {!r}'.format(group_name,
                                                                  var_name,
                                                                  item,
                                                                  code))
            code_lines += ['']
            # The run statement
            code_lines.append('run(%r, namespace=%r)' % (duration, namespace))
            code_lines += ['']

        # Simply print the code to the screen for now
        print '\n'.join(code_lines)


# Helper functions to convert the objects into string descriptions

def eq_string(equations):
    lines = []
    for eq in equations.ordered:
        unit = '1' if eq.unit == Unit(1) else repr(eq.unit)
        flags = ''
        if len(eq.flags):
            flags = '({})'.format(', '.join(eq.flags))
        if eq.type == DIFFERENTIAL_EQUATION:
            lines.append('d{eq.varname}/dt = {eq.expr} : {unit} {flags}'.format(eq=eq,
                                                                                unit=unit,
                                                                                flags=flags))
        elif eq.type == SUBEXPRESSION:
            lines.append('{eq.varname} = {eq.expression} : {unit} {flags}'.format(eq=eq,
                                                                                  unit=unit,
                                                                                  flags=flags))
        elif eq.type == PARAMETER:
            lines.append('{eq.varname} : {unit} {flags}'.format(eq=eq, unit=unit,
                                                                flags=flags))
    return '\n'.join(lines)


def get_namespace_dict(identifiers, neurongroup, run_namespace):
    variables = neurongroup.resolve_all(identifiers, run_namespace)
    namespace = {key: Quantity(value.get_value(),
                               dim=value.unit.dimensions)
                 for key, value in variables.iteritems()
                 if (isinstance(value, Constant) and
                     not key in DEFAULT_UNITS)}
    return namespace


def description(brian_obj, run_namespace):
    if isinstance(brian_obj, NeuronGroup):
        return neurongroup_description(brian_obj, run_namespace)
    elif isinstance(brian_obj, SpikeMonitor):
        desc = '%s, name=%r' % (brian_obj.source.name, brian_obj.name)
        return '%s = SpikeMonitor(%s)' % (brian_obj.name, desc), {}
    else:
        return '', {}


def neurongroup_description(neurongroup, run_namespace):
    eqs = eq_string(neurongroup.user_equations)
    identifiers = neurongroup.user_equations.identifiers
    desc = "%d,\n'''%s'''" % (len(neurongroup), eqs)
    if 'spike' in neurongroup.events:
        threshold = neurongroup.events['spike']
        desc += ',\nthreshold=%r' % threshold
        identifiers |= get_identifiers(threshold)
    if 'spike' in neurongroup.event_codes:
        reset = neurongroup.event_codes['spike']
        desc += ',\nreset=%r' % reset
        identifiers |= get_identifiers(reset)
    if neurongroup._refractory is not None:
        refractory = neurongroup._refractory
        desc += ',\nrefractory=%r' % refractory
        if isinstance(refractory, basestring):
            identifiers |= get_identifiers(refractory)
    namespace = get_namespace_dict(identifiers, neurongroup,
                                   run_namespace)
    desc += ',\nname=%r' % neurongroup.name
    desc = '%s = NeuronGroup(%s)' % (neurongroup.name, desc)
    return desc, namespace


# Make the device known
example_device = ExampleDevice()
all_devices['example'] = example_device
