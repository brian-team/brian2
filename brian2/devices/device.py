'''
Module containing the `Device` base class as well as the `RuntimeDevice`
implementation and some helper functions to access/set devices.
'''
from weakref import WeakKeyDictionary
import numbers

import numpy as np

from brian2.memory.dynamicarray import DynamicArray, DynamicArray1D
from brian2.codegen.targets import codegen_targets
from brian2.codegen.runtime.numpy_rt import NumpyCodeObject
from brian2.core.names import find_name
from brian2.core.preferences import prefs
from brian2.core.variables import ArrayVariable, DynamicArrayVariable
from brian2.core.functions import Function
from brian2.units import ms
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import code_representation, indent

__all__ = ['Device', 'RuntimeDevice',
           'get_device', 'set_device',
           'all_devices', 'reinit_devices',
           'reset_device', 'device', 'seed'
           ]

logger = get_logger(__name__)

all_devices = {}


prefs.register_preferences('devices', 'Device preferences')


#: caches the automatically determined code generation target
_auto_target = None

def auto_target():
    '''
    Automatically chose a code generation target (invoked when the
    `codegen.target` preference is set to `'auto'`. Caches its result so it
    only does the check once. Prefers weave > cython > numpy.

    Returns
    -------
    target : class derived from `CodeObject`
        The target to use
    '''
    global _auto_target
    if _auto_target is None:
        target_dict = dict((target.class_name, target)
                           for target in codegen_targets
                           if target.class_name)
        using_fallback = False
        if 'weave' in target_dict and target_dict['weave'].is_available():
            _auto_target = target_dict['weave']
        elif 'cython' in target_dict and target_dict['cython'].is_available():
            _auto_target = target_dict['cython']
        else:
            _auto_target = target_dict['numpy']
            using_fallback = True

        if using_fallback:
            logger.info('Cannot use compiled code, falling back to the numpy '
                        'code generation target. Note that this will likely '
                        'be slower than using compiled code. Set the code '
                        'generation to numpy manually to avoid this message:\n'
                        'prefs.codegen.target = "numpy"',
                        'codegen_fallback', once=True)
        else:
            logger.debug(('Chosing %r as the code generation '
                         'target.') % _auto_target.class_name)

    return _auto_target

class Device(object):
    '''
    Base Device object.
    '''
    def __init__(self):
        #: The network schedule that this device supports. If the device only
        #: supports a specific, fixed schedule, it has to set this attribute to
        #: the respective schedule (see `Network.schedule` for details). If it
        #: supports arbitrary schedules, it should be set to ``None`` (the
        #: default).
        self.network_schedule = None

        self.defaultclock = None

        self._maximum_run_time = None

        self._state_tuple = (self.__module__, self.__class__.__name__)

    def _set_maximum_run_time(self, maximum_run_time):
        '''
        Sets a maximum time for a run before it will break. Used primarily for testing purposes. Not guaranteed to be
        respected by a device.
        '''
        self._maximum_run_time = maximum_run_time

    def get_array_name(self, var, access_data=True):
        '''
        Return a globally unique name for `var`.

        Parameters
        ----------
        access_data : bool, optional
            For `DynamicArrayVariable` objects, specifying `True` here means the
            name for the underlying data is returned. If specifying `False`,
            the name of object itself is returned (e.g. to allow resizing).

        Returns
        -------
        name : str
            The name for `var`.
        '''
        raise NotImplementedError()

    def get_len(self, array):
        '''
        Return the length of the array.

        Parameters
        ----------
        array : `ArrayVariable`
            The array for which the length is requested.

        Returns
        -------
        l : int
            The length of the array.
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

    def init_with_zeros(self, var, dtype):
        '''
        Initialize an array with zeros.

        Parameters
        ----------
        var : `ArrayVariable`
            The array to initialize with zeros.
        dtype : `dtype`
            The data type to use for the array.
        '''
        raise NotImplementedError()

    def init_with_arange(self, var, start, dtype):
        '''
        Initialize an array with an integer range.

        Parameters
        ----------
        var : `ArrayVariable`
            The array to fill with the integer range.
        start : int
            The start value for the integer range
        dtype : `dtype`
            The data type to use for the array.
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

    def resize(self, var, new_size):
        '''
        Resize a `DynamicArrayVariable`.

        Parameters
        ----------
        var : `DynamicArrayVariable`
            The variable that should be resized.
        new_size : int
            The new size of the variable
        '''
        raise NotImplementedError()

    def resize_along_first(self, var, new_size):
        # Can be overwritten with a better implementation
        return self.resize(var, new_size)

    def seed(self, seed=None):
        '''
        Set the seed for the random number generator.

        Parameters
        ----------
        seed : int, optional
            The seed value for the random number generator, or ``None`` (the
            default) to set a random seed.
        '''
        raise NotImplementedError()

    def code_object_class(self, codeobj_class=None, fallback_pref='codegen.target'):
        '''
        Return `CodeObject` class according to input/default settings

        Parameters
        ----------
        codeobj_class : a `CodeObject` class, optional
            If this is keyword is set to None or no arguments are given, this method will return
            the default.
        fallback_pref : str, optional
            String describing which attribute of prefs to access to retrieve the 'default' target.
            Usually this is codegen.target, but in some cases we want to use object-specific targets
            such as codegen.string_expression_target.

        Returns
        -------
        codeobj_class : class
            The `CodeObject` class that should be used
        '''
        if isinstance(codeobj_class, str):
            raise TypeError("codeobj_class argument given to code_object_class device method "
                            "should be a CodeObject class, not a string. You can, however, "
                            "send a string description of the target desired for the CodeObject "
                            "under the keyword fallback_pref")
        if codeobj_class is None:
            codeobj_class = prefs[fallback_pref]
            if isinstance(codeobj_class, str):
                if codeobj_class == 'auto':
                    return auto_target()
                for target in codegen_targets:
                    if target.class_name == codeobj_class:
                        return target
                # No target found
                targets = ['auto'] + [target.class_name
                                      for target in codegen_targets
                                      if target.class_name]
                raise ValueError("Unknown code generation target: %s, should be "
                                 " one of %s" % (codeobj_class, targets))
        else:
            return codeobj_class

    def code_object(self, owner, name, abstract_code, variables, template_name,
                    variable_indices, codeobj_class=None,
                    template_kwds=None, override_conditional_write=None):
        name = find_name(name)
        codeobj_class = self.code_object_class(codeobj_class)
        template = getattr(codeobj_class.templater, template_name)
        iterate_all = template.iterate_all
        generator = codeobj_class.generator_class(variables=variables,
                                                  variable_indices=variable_indices,
                                                  owner=owner,
                                                  iterate_all=iterate_all,
                                                  codeobj_class=codeobj_class,
                                                  override_conditional_write=override_conditional_write,
                                                  allows_scalar_write=template.allows_scalar_write,
                                                  name=name,
                                                  template_name=template_name)
        if template_kwds is None:
            template_kwds = dict()
        else:
            template_kwds = template_kwds.copy()

        # Check that all functions are available
        for varname, value in variables.iteritems():
            if isinstance(value, Function):
                try:
                    value.implementations[codeobj_class]
                except KeyError as ex:
                    # if we are dealing with numpy, add the default implementation
                    if codeobj_class is NumpyCodeObject:
                        value.implementations.add_numpy_implementation(value.pyfunc)
                    else:
                        raise NotImplementedError(('Cannot use function '
                                                   '%s: %s') % (varname, ex))

        logger.diagnostic('%s abstract code:\n%s' % (name, indent(code_representation(abstract_code))))

        scalar_code, vector_code, kwds = generator.translate(abstract_code,
                                                             dtype=prefs['core.default_float_dtype'])
        # Add the array names as keywords as well
        for varname, var in variables.iteritems():
            if isinstance(var, ArrayVariable):
                pointer_name = generator.get_array_name(var)
                if var.scalar:
                    pointer_name += '[0]'
                template_kwds[varname] = pointer_name
                if hasattr(var, 'resize'):
                    dyn_array_name = generator.get_array_name(var,
                                                              access_data=False)
                    template_kwds['_dynamic_'+varname] = dyn_array_name


        template_kwds.update(kwds)
        logger.diagnostic('%s snippet (scalar):\n%s' % (name, indent(code_representation(scalar_code))))
        logger.diagnostic('%s snippet (vector):\n%s' % (name, indent(code_representation(vector_code))))

        code = template(scalar_code, vector_code,
                        owner=owner, variables=variables, codeobj_name=name,
                        variable_indices=variable_indices,
                        get_array_name=generator.get_array_name,
                        **template_kwds)
        logger.diagnostic('%s code:\n%s' % (name, indent(code_representation(code))))

        codeobj = codeobj_class(owner, code, variables, variable_indices,
                                template_name=template_name,
                                template_source=template.template_source,
                                name=name)
        codeobj.compile()
        return codeobj
    
    def activate(self, build_on_run=True, **kwargs):
        '''
        Called when this device is set as the current device.
        '''
        from brian2.core.clocks import Clock  # avoid import issues

        if self.defaultclock is None:
            self.defaultclock = Clock(dt=0.1*ms, name='defaultclock')
        self._set_maximum_run_time(None)
        self.build_on_run = build_on_run
        self.build_options = dict(kwargs)

    def insert_device_code(self, slot, code):
        # Deprecated
        raise AttributeError("The method 'insert_device_code' has been renamed "
                             "to 'insert_code'.")

    def insert_code(self, slot, code):
        '''
        Insert code directly into a given slot in the device. By default does nothing.
        '''
        logger.warn("Ignoring device code, unknown slot: %s, code: %s" % (slot, code))
        
    def build(self, **kwds):
        '''
        For standalone projects, called when the project is ready to be built. Does nothing for runtime mode.
        '''
        pass

    def reinit(self):
        '''
        Reinitialize the device. For standalone devices, clears all the internal
        state of the device.
        '''
        pass


class RuntimeDevice(Device):
    '''
    The default device used in Brian, state variables are stored as numpy
    arrays in memory.
    '''
    def __init__(self):
        super(RuntimeDevice, self).__init__()
        #: Mapping from `Variable` objects to numpy arrays (or `DynamicArray`
        #: objects). Arrays in this dictionary will disappear as soon as the
        #: last reference to the `Variable` object used as a key is gone
        self.arrays = WeakKeyDictionary()
        # Note that the buffers only store a pointer to the actual random
        # numbers -- the buffer will be filled in weave/Cython code
        self.randn_buffer = np.zeros(1, dtype=np.intp)
        self.rand_buffer = np.zeros(1, dtype=np.intp)
        self.randn_buffer_index = np.zeros(1, dtype=np.int32)
        self.rand_buffer_index = np.zeros(1, dtype=np.int32)

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
            if var.ndim == 1:
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

    def resize_along_first(self, var, new_size):
        self.arrays[var].resize_along_first(new_size)

    def init_with_zeros(self, var, dtype):
        self.arrays[var][:] = 0

    def init_with_arange(self, var, start, dtype):
        self.arrays[var][:] = np.arange(start, stop=var.get_len()+start,
                                        dtype=dtype)

    def fill_with_array(self, var, arr):
        self.arrays[var][:] = arr

    def spike_queue(self, source_start, source_end):
        # Use the C++ version of the SpikeQueue when available
        try:
            from brian2.synapses.cythonspikequeue import SpikeQueue
            logger.diagnostic('Using the C++ SpikeQueue', once=True)
        except ImportError:
            from brian2.synapses.spikequeue import SpikeQueue
            logger.diagnostic('Using the Python SpikeQueue', once=True)

        return SpikeQueue(source_start=source_start, source_end=source_end)

    def seed(self, seed=None):
        '''
        Set the seed for the random number generator.

        Parameters
        ----------
        seed : int, optional
            The seed value for the random number generator, or ``None`` (the
            default) to set a random seed.
        '''
        np.random.seed(seed)
        self.rand_buffer_index[:] = 0
        self.randn_buffer_index[:] = 0


class Dummy(object):
    '''
    Dummy object
    '''
    def __getattr__(self, name):
        return Dummy()
    def __call__(self, *args, **kwds):
        return Dummy()
    def __enter__(self):
        return Dummy()
    def __exit__(self, type, value, traceback):
        pass
    def __getitem__(self, i):
        return Dummy()
    def __setitem__(self, i, val):
        pass


class CurrentDeviceProxy(object):
    '''
    Method proxy for access to the currently active device
    '''
    def __getattr__(self, name):
        if not hasattr(active_device, name):
            if name.startswith('_'):
                # Do not fake private/magic attributes
                raise AttributeError(('Active device does not have an '
                                      'attribute %s') % name)
            else:
                logger.warn(("Active device does not have an attribute '%s', "
                             "ignoring this") % name)
                attr = Dummy()
        else:
            attr = getattr(active_device, name)
        return attr

#: Proxy object to access methods of the current device
device = CurrentDeviceProxy()
#: The currently active device (set with `set_device`)
active_device = None


def get_device():
    '''
    Gets the actve `Device` object
    '''
    global active_device
    return active_device

#: A stack of previously set devices as a tuple with their options (see
#: `set_device`): (device, build_on_run, build_options)
previous_devices = []


def set_device(device, build_on_run=True, **kwargs):
    '''
    Set the device used for simulations.

    Parameters
    ----------
    device : `Device` or str
        The `Device` object or the name of the device.
    build_on_run : bool, optional
        Whether a call to `run` (or `Network.run`) should directly trigger a
        `Device.build`. This is only relevant for standalone devices and means
        that a run call directly triggers the start of a simulation. If the
        simulation consists of multiple run calls, set ``build_on_run`` to
        ``False`` and call `Device.build` explicitly. Defaults to ``True``.
    kwargs : dict, optional
        Only relevant when ``build_on_run`` is ``True``: additional arguments
        that will be given to the `Device.build` call.
    '''
    global previous_devices
    if active_device is not None:
        prev_build_on_run = getattr(active_device, 'build_on_run', True)
        prev_build_options = getattr(active_device, 'build_options', {})
        previous_devices.append((active_device,
                                 prev_build_on_run,
                                 prev_build_options))
    _do_set_device(device, build_on_run, **kwargs)


def _do_set_device(device, build_on_run=True, **kwargs):
    global active_device

    if isinstance(device, str):
        device = all_devices[device]
    if active_device is not None and active_device.defaultclock is not None:
        previous_dt = active_device.defaultclock.dt
    else:
        previous_dt = None
    active_device = device

    active_device.activate(build_on_run=build_on_run, **kwargs)
    if previous_dt is not None:
        # Copy over the dt information of the defaultclock
        active_device.defaultclock.dt = previous_dt


def reset_device(device=None):
    '''
    Reset to a previously used device. Restores also the previously specified
    build options (see `set_device`) for the device. Mostly useful for internal
    Brian code and testing on various devices.

    Parameters
    ----------
    device : `Device` or str, optional
        The device to go back to. If none is specified, go back to the device
        chosen with `set_device` before the current one.
    '''
    global previous_devices
    if isinstance(device, str):
        device = all_devices[device]

    if len(previous_devices) == 0 and device is None:
        device = runtime_device
        build_on_run = True
        build_options = {}
    elif device is None:
        device, build_on_run, build_options = previous_devices.pop()
    else:
        build_on_run = device.build_on_run
        build_options = device.build_options

    _do_set_device(device, build_on_run, **build_options)


def reinit_devices():
    '''
    Reinitialize all devices, call `Device.activate` again on the current
    device and reset the preferences. Used as a "teardown" function in testing,
    if users want to reset their device (e.g. for multiple standalone runs in a
    single script), calling ``device.reinit()`` followed by
    ``device.activate()`` should normally be sufficient.

    Notes
    -----
    This also resets the `defaultclock`, i.e. a non-standard ``dt`` has to be
    set again.
    '''
    from brian2 import restore_initial_state  # avoids circular import

    for device in all_devices.itervalues():
        device.reinit()

    if active_device is not None:
        # Reactivate the current device
        reset_device(active_device)

    restore_initial_state()


def seed(seed=None):
    '''
    Set the seed for the random number generator.

    Parameters
    ----------
    seed : int, optional
        The seed value for the random number generator, or ``None`` (the
        default) to set a random seed.

    Notes
    -----
    This function delegates the call to `Device.seed` of the current device.
    '''
    if seed is not None and not isinstance(seed, numbers.Integral):
        raise TypeError('Seed has to be None or an integer, was '
                        '%s' % type(seed))
    get_device().seed(seed)


runtime_device = RuntimeDevice()
all_devices['runtime'] = runtime_device

