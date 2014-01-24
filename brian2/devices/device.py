'''
Module containing the `Device` base class as well as the `RuntimeDevice`
implementation and some helper functions to access/set devices.
'''

import numpy as np

from brian2.memory.dynamicarray import DynamicArray, DynamicArray1D
from brian2.codegen.targets import codegen_targets
from brian2.codegen.functions import add_numpy_implementation
from brian2.codegen.codeobject import create_runner_codeobj, check_code_units
from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
from brian2.codegen.translation import translate
from brian2.core.namespace import get_local_namespace
from brian2.core.names import find_name
from brian2.core.preferences import brian_prefs
from brian2.core.variables import (Variables, ArrayVariable, DynamicArrayVariable,
                                   Subexpression)
from brian2.core.functions import Function
from brian2.units.fundamentalunits import Unit, fail_for_dimension_mismatch
from brian2.utils.logger import get_logger

__all__ = ['Device', 'RuntimeDevice',
           'get_device', 'set_device',
           'all_devices',
           'insert_device_code',
           'device_override',
           'build',
           ]

logger = get_logger(__name__)

all_devices = {}


def get_default_codeobject_class():
    '''
    Returns the default `CodeObject` class from the preferences.
    '''
    codeobj_class = brian_prefs['codegen.target']
    if isinstance(codeobj_class, str):
        for target in codegen_targets:
            if target.class_name == codeobj_class:
                return target
        # No target found
        raise ValueError("Unknown code generation target: %s, should be "
                         " one of %s"%(codeobj_class,
                                       [target.class_name
                                        for target in codegen_targets]))
    return codeobj_class


class Device(object):
    '''
    Base Device object.
    '''
    def __init__(self):
        pass

    def get_array_name(self, var, access_data=True):
        '''
        Return a globally unique name for `var`.

        Parameters
        ----------
        access_data : bool, optional
            For `DynamicArrayVariable` objects, specifying `True` here means the
            name for the underlying data is returned. If specifying `False`,
            the name of object itself is returned (e.g. to allow resizing).
        '''
        raise NotImplementedError()

    def add_array(self, var):
        '''
        Add an array to this device.

        Parameters
        ----------
        var : `ArrayVariable`
            The array to add.
        '''
        raise NotImplementedError()

    def init_with_zeros(self, var):
        '''
        Initialize an array with zeros.

        Parameters
        ----------
        var : `ArrayVariable`
            The array to initialize with zeros.
        '''
        raise NotImplementedError()

    def init_with_arange(self, var, start):
        '''
        Initialize an array with an integer range.

        Parameters
        ----------
        var : `ArrayVariable`
            The array to fill with the integer range.
        start : int
            The start value for the integer range
        '''
        raise NotImplementedError()

    def fill_with_array(self, var, arr):
        '''
        Fill an array with the values given in another array.

        Parameters
        ----------
        var : `ArrayVariable`
            The array to fill.
        arr : `ndarray`
            The array values that should be copied to `var`.
        '''
        raise NotImplementedError()

    def get_with_index_array(self, group, variable_name, variable, item):
        '''
        Gets a variable using array indices. Is called by
        `VariableView.get_item` for statements such as ``print G.v[[0, 1, 2]]``

        Parameters
        ----------
        group : `Group`
            The group providing the context for the indexing.
        variable_name : str
            The name of the variable in its context (e.g. ``'g_post'`` for a
            variable with name ``'g'``)
        variable : `ArrayVariable`
            The `ArrayVariable` object for the variable to be set
        item : `ndarray`
            The indices for the variable (in the context of this `group`).
        '''
        raise NotImplementedError(('This device does not support accessing '
                                   'a state variable with an index array or '
                                   'slice.'))

    def get_with_expression(self, group, variable_name, variable, code, level=0):
        '''
        Gets a variable using a string expression. Is called by
        `VariableView.get_item` for statements such as
        ``print G.v['g_syn > 0']``.

        Parameters
        ----------
        group : `Group`
            The group providing the context for the indexing.
        variable_name : str
            The name of the variable in its context (e.g. ``'g_post'`` for a
            variable with name ``'g'``)
        variable : `ArrayVariable`
            The `ArrayVariable` object for the variable to be set
        code : str
            An expression that states a condition for elements that should be
            selected. Can contain references to indices, such as ``i`` or ``j``
            and to state variables. For example: ``'i>3 and v>0*mV'``.
        level : int, optional
            How much farther to go up in the stack to find the namespace.
        '''
        # interpret the string expression
        namespace = get_local_namespace(level+1)
        additional_namespace = ('implicit-namespace', namespace)
        # Add the recorded variable under a known name to the variables
        # dictionary. Important to deal correctly with
        # the type of the variable in C++
        variables = Variables(None)
        variables.add_auxiliary_variable('_variable', unit=variable.unit,
                                         dtype=variable.dtype,
                                         scalar=variable.scalar,
                                         is_bool=variable.is_bool)
        variables.add_auxiliary_variable('_cond', unit=Unit(1), dtype=np.bool,
                                         is_bool=True)

        abstract_code = '_variable = ' + variable_name + '\n'
        abstract_code += '_cond = ' + code
        check_code_units(abstract_code, group,
                         additional_namespace=additional_namespace,
                         additional_variables=variables)
        codeobj = create_runner_codeobj(group,
                                        abstract_code,
                                        'group_variable_get_conditional',
                                        additional_variables=variables,
                                        additional_namespace=additional_namespace,
                                        )
        return codeobj()

    def set_with_index_array(self, group, variable_name, variable, item, value,
                             check_units):
        '''
        Sets a variable using array indices. Is called by
        `VariableView.set_item` for statements such as ``S.var[:, :] = 42``.

        Parameters
        ----------
        group : `Group`
            The group providing the context for the indexing.
        variable_name : str
            The name of the variable to be set.
        variable : `ArrayVariable`
            The variable to be set.
        item : `ndarray`
            The indices for the variable (in the context of this `group`).
        value : `ndarray`
            Array containing the target values. Has to be the same size as
            `group_indices`.
        check_units : bool
            Whether to check the unit of `value` for consistency.
        '''
        raise NotImplementedError(('This device does not support setting '
                                   'a state variable with an index array or '
                                   'slice.'))

    def set_with_expression(self, group, varname, variable, item, code,
                            check_units=True, level=0):
        '''
        Sets a variable using a string expression. Is called by
        `VariableView.set_item` for statements such as
        ``S.var[:, :] = 'exp(-abs(i-j)/space_constant)*nS'``

        Parameters
        ----------
        group : `Group`
            The group providing the context for the indexing.
        varname : str
            The name of the variable to be set
        variable : `ArrayVariable`
            The `ArrayVariable` object for the variable to be set.
        item : `ndarray`
            The indices for the variable (in the context of this `group`).
        code : str
            The code that should be executed to set the variable values.
            Can contain references to indices, such as `i` or `j`
        check_units : bool, optional
            Whether to check the units of the expression.
        level : int, optional
            How much farther to go up in the stack to find the namespace.
        '''
        indices = group.calc_indices(item)
        abstract_code = varname + ' = ' + code
        namespace = get_local_namespace(level + 1)
        additional_namespace = ('implicit-namespace', namespace)
        variables = Variables(None)
        variables.add_array('_group_idx', unit=Unit(1),
                            size=len(indices), dtype=np.int32)
        variables['_group_idx'].set_value(indices)

        # TODO: Have an additional argument to avoid going through the index
        # array for situations where iterate_all could be used
        codeobj = create_runner_codeobj(group,
                                 abstract_code,
                                 'group_variable_set',
                                 additional_variables=variables,
                                 additional_namespace=additional_namespace,
                                 check_units=check_units)
        codeobj()

    def set_with_expression_conditional(self, group, varname, variable, cond,
                                        code, check_units=True, level=0):
        '''
        Sets a variable using a string expression and string condition. Is
        called by `VariableView.set_item` for statements such as
        ``S.var['i!=j'] = 'exp(-abs(i-j)/space_constant)*nS'``

        Parameters
        ----------
        group : `Group`
            The group providing the context for the indexing.
        varname : str
            The name of the variable to be set.
        variable : `ArrayVariable`
            The `ArrayVariable` object for the variable to be set.
        cond : str
            The string condition for which the variables should be set.
        code : str
            The code that should be executed to set the variable values.
        check_units : bool, optional
            Whether to check the units of the expression.
        level : int, optional
            How much farther to go up in the stack to find the namespace.
        '''

        abstract_code_cond = '_cond = '+cond
        abstract_code = varname + ' = ' + code
        namespace = get_local_namespace(level + 1)
        additional_namespace = ('implicit-namespace', namespace)
        variables = Variables(None)
        variables.add_auxiliary_variable('_cond', unit=Unit(1), dtype=np.bool,
                                         is_bool=True)
        check_code_units(abstract_code_cond, group,
                         additional_variables=variables,
                         additional_namespace=additional_namespace)
        # TODO: Have an additional argument to avoid going through the index
        # array for situations where iterate_all could be used
        codeobj = create_runner_codeobj(group,
                                 {'condition': abstract_code_cond,
                                  'statement': abstract_code},
                                 'group_variable_set_conditional',
                                 additional_variables=variables,
                                 additional_namespace=additional_namespace,
                                 check_units=check_units)
        codeobj()

    def spike_queue(self, source_start, source_end):
        '''
        Create and return a new `SpikeQueue` for this `Device`.

        Parameters
        ----------
        source_start : int
            The start index of the source group (necessary for subgroups)
        source_end : int
            The end index of the source group (necessary for subgroups)
        '''
        raise NotImplementedError()

    def code_object_class(self, codeobj_class=None):
        if codeobj_class is None:
            codeobj_class = get_default_codeobject_class()
        return codeobj_class

    def code_object(self, owner, name, abstract_code, variables, template_name,
                    variable_indices, codeobj_class=None,
                    template_kwds=None):
        codeobj_class = self.code_object_class(codeobj_class)
        language = codeobj_class.language

        if template_kwds is None:
            template_kwds = dict()
        else:
            template_kwds = template_kwds.copy()

        template = getattr(codeobj_class.templater, template_name)

        # Check that all functions are available
        for varname, value in variables.iteritems():
            if isinstance(value, Function):
                try:
                    value.implementations[codeobj_class]
                except KeyError as ex:
                    # if we are dealing with numpy, add the default implementation
                    if codeobj_class is NumpyCodeObject:
                        add_numpy_implementation(value, value.pyfunc)
                    else:
                        raise NotImplementedError(('Cannot use function '
                                                   '%s: %s') % (varname, ex))

        if isinstance(abstract_code, dict):
            for k, v in abstract_code.items():
                logger.debug('%s abstract code key %s:\n%s' % (name, k, v))
        else:
            logger.debug(name + " abstract code:\n" + abstract_code)
        iterate_all = template.iterate_all
        snippet, kwds = translate(abstract_code, variables,
                                  dtype=brian_prefs['core.default_scalar_dtype'],
                                  codeobj_class=codeobj_class,
                                  variable_indices=variable_indices,
                                  iterate_all=iterate_all)
        # Add the array names as keywords as well
        for varname, var in variables.iteritems():
            if isinstance(var, ArrayVariable):
                pointer_name = language.get_array_name(var)
                template_kwds[varname] = pointer_name
                if hasattr(var, 'resize'):
                    dyn_array_name = language.get_array_name(var,
                                                             access_data=False)
                    template_kwds['_dynamic_'+varname] = dyn_array_name


        template_kwds.update(kwds)
        logger.debug(name + " snippet:\n" + str(snippet))

        name = find_name(name)

        code = template(snippet,
                        owner=owner, variables=variables, codeobj_name=name,
                        variable_indices=variable_indices,
                        get_array_name=language.get_array_name,
                        **template_kwds)
        logger.debug(name + " code:\n" + str(code))

        codeobj = codeobj_class(owner, code, variables, name=name)
        codeobj.compile()
        return codeobj
    
    def activate(self):
        '''
        Called when this device is set as the current device.
        '''
        pass

    def insert_device_code(self, slot, code):
        '''
        Insert code directly into a given slot in the device. By default does nothing.
        '''
        logger.warn("Ignoring device code, unknown slot: %s, code: %s" % (slot, code))
        
    def build(self, **kwds):
        '''
        For standalone projects, called when the project is ready to be built. Does nothing for runtime mode.
        '''
        pass
    
    
class RuntimeDevice(Device):
    '''
    '''
    def __init__(self):
        super(Device, self).__init__()
        #: Mapping from `Variable` objects to numpy arrays (or `DynamicArray`
        #: objects)
        self.arrays = {}
        
    def get_array_name(self, var, access_data=True):
        # if no owner is set, this is a temporary object (e.g. the array
        # of indices when doing G.x[indices] = ...). The name is not
        # necessarily unique over several CodeObjects in this case.
        owner_name = getattr(var.owner, 'name', 'temporary')

        if isinstance(var, DynamicArrayVariable):
            if access_data:
                return '_array_' + owner_name + '_' + var.name
            else:
                return '_dynamic_array_' + owner_name + '_' + var.name
        elif isinstance(var, ArrayVariable):
            return '_array_' + owner_name + '_' + var.name
        else:
            raise TypeError(('Do not have a name for variable of type '
                             '%s') % type(var))

    def add_array(self, var):
        # This creates the actual numpy arrays (or DynamicArrayVariable objects)
        if isinstance(var, DynamicArrayVariable):
            if var.dimensions == 1:
                arr = DynamicArray1D(var.size, dtype=var.dtype)
            else:
                arr = DynamicArray(var.size, dtype=var.dtype)
        else:
            arr = np.empty(var.size, dtype=var.dtype)

        self.arrays[var] = arr

    def get_value(self, var, access_data=True):
        if isinstance(var, DynamicArrayVariable) and access_data:
                return self.arrays[var].data
        else:
            return self.arrays[var]

    def set_value(self, var, value):
        self.arrays[var][:] = value

    def resize(self, var, new_size):
        self.arrays[var].resize(new_size)

    def get_with_index_array(self, group, variable_name, variable, item):
        if variable.scalar:
            if not ((isinstance(item, slice) and item == slice(None)) or item == 0 or (hasattr(item, '__len__')
                                                                                           and len(item) == 0)):
                raise IndexError('Variable is a scalar variable.')
            indices = np.array([0])
        else:
            indices = group.calc_indices(item)

        # For "normal" variables, we can directly access the underlying data
        # and use the usual slicing syntax. For subexpressions, however, we
        # have to evaluate code for the given indices
        if isinstance(variable, Subexpression):
            variables = Variables(None)
            variables.add_auxiliary_variable('_variable', unit=variable.unit,
                                             dtype=variable.dtype,
                                             scalar=variable.scalar,
                                             is_bool=variable.is_bool)
            variables.add_array('_group_idx', unit=Unit(1),
                                size=len(indices), dtype=np.int32)
            variables['_group_idx'].set_value(indices)

            abstract_code = '_variable = ' + variable_name + '\n'
            codeobj = create_runner_codeobj(group,
                                            abstract_code,
                                            'group_variable_get',
                                            additional_variables=variables
            )
            return codeobj()
        else:
            # We are not going via code generation so we have to take care
            # of correct indexing (in particular for subgroups) explicitly
            var_index = group.variables.indices[variable_name]
            if var_index != '_idx':
                indices = group.variables[var_index].get_value()[indices]
            return variable.get_value()[indices]

    def set_with_index_array(self, group, variable_name, variable, item, value,
                        check_units):
        if variable.scalar:
            if not ((isinstance(item, slice) and item == slice(None)) or item == 0 or (hasattr(item, '__len__')
                                                                                           and len(item) == 0)):
                raise IndexError('Variable is a scalar variable.')
            indices = np.array([0])
        else:
            indices = group.calc_indices(item)
        # We are not going via code generation so we have to take care
        # of correct indexing (in particular for subgroups) explicitly
        var_index = group.variables.indices[variable_name]
        if var_index != '_idx':
            indices = group.variables[var_index].get_value()[indices]

        if check_units:
            fail_for_dimension_mismatch(variable.unit, value,
                                        'Incorrect unit for setting variable %s' % variable_name)

        variable.get_value()[indices] = value

    def init_with_zeros(self, var):
        self.arrays[var][:] = 0

    def init_with_arange(self, var, start):
        self.arrays[var][:] = np.arange(start, stop=var.size+start)

    def fill_with_array(self, var, arr):
        self.arrays[var][:] = arr

    def spike_queue(self, source_start, source_end):
        # Use the C++ version of the SpikeQueue when available
        try:
            from brian2.synapses.cythonspikequeue import SpikeQueue
            logger.info('Using the C++ SpikeQueue', once=True)
        except ImportError:
            from brian2.synapses.spikequeue import SpikeQueue
            logger.info('Using the Python SpikeQueue', once=True)

        return SpikeQueue(source_start=source_start, source_end=source_end)


runtime_device = RuntimeDevice()

all_devices['runtime'] = runtime_device

current_device = runtime_device

def get_device():
    '''
    Gets the current `Device` object
    '''
    global current_device
    return current_device

def set_device(device):
    '''
    Sets the current `Device` object
    '''
    global current_device
    if isinstance(device, str):
        device = all_devices[device]
    current_device = device
    current_device.activate()


def insert_device_code(slot, code):
    '''
    Inserts the given set of code into the slot defined by the device.
    
    The behaviour of this function is device dependent. The runtime device ignores it (useful for debugging).
    '''
    get_device().insert_device_code(slot, code)


def device_override(name):
    '''
    Decorates a function/method to allow it to be overridden by the current `Device`.
    
    The ``name`` is the function name in the `Device` to use as an override if it exists.
    '''
    def device_override_decorator(func):
        def device_override_decorated_function(*args, **kwds):
            curdev = get_device()
            if hasattr(curdev, name):
                return getattr(curdev, name)(*args, **kwds)
            else:
                return func(*args, **kwds)
        
        device_override_decorated_function.__doc__ = func.__doc__
        
        return device_override_decorated_function
    
    return device_override_decorator


def build(**kwds):
    '''
    Builds the project for standalone devices, does nothing for runtime. Calls `Device.build`.
    '''
    get_device().build(**kwds)
