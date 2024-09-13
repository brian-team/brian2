"""
All Brian objects should derive from `BrianObject`.
"""

import functools
import os
import sys
import traceback
import weakref

from brian2.core.names import Nameable
from brian2.units.allunits import second
from brian2.units.fundamentalunits import check_units
from brian2.utils.logger import get_logger

__all__ = [
    "BrianObject",
    "BrianObjectException",
]

logger = get_logger(__name__)


class BrianObject(Nameable):
    """
    All Brian objects derive from this class, defines magic tracking and update.

    See the documentation for `Network` for an explanation of which
    objects get updated in which order.

    Parameters
    ----------
    dt : `Quantity`, optional
        The time step to be used for the simulation. Cannot be combined with
        the `clock` argument.
    clock : `Clock`, optional
        The update clock to be used. If neither a clock, nor the `dt` argument
        is specified, the `defaultclock` will be used.
    when : str, optional
        In which scheduling slot to simulate the object during a time step.
        Defaults to ``'start'``. See :ref:`scheduling` for possible values.
    order : int, optional
        The priority of this object for operations occurring at the same time
        step and in the same scheduling slot. Defaults to 0.
    namespace: dict, optional
        A dictionary mapping identifier names to objects. If not given, the
        namespace will be filled in at the time of the call of `Network.run`,
        with either the values from the ``namespace`` argument of the
        `Network.run` method or from the local context, if no such argument is
        given.
    name : str, optional
        A unique name for the object - one will be assigned automatically if
        not provided (of the form ``brianobject_1``, etc.).
    Notes
    -----

    The set of all `BrianObject` objects is stored in ``BrianObject.__instances__()``.
    """

    @check_units(dt=second)
    def __init__(
        self,
        dt=None,
        clock=None,
        when="start",
        order=0,
        namespace=None,
        name="brianobject*",
    ):
        # Setup traceback information for this object
        creation_stack = []
        bases = []
        for modulename in ["brian2"]:
            if modulename in sys.modules:
                base, _ = os.path.split(sys.modules[modulename].__file__)
                bases.append(base)
        for fname, linenum, funcname, line in traceback.extract_stack():
            if all(base not in fname for base in bases):
                s = f"  File '{fname}', line {linenum}, in {funcname}\n    {line}"
                creation_stack.append(s)
        creation_stack = [""] + creation_stack
        #: A string indicating where this object was created (traceback with any parts of Brian code removed)
        self._creation_stack = creation_stack[-1]
        self._full_creation_stack = "\n".join(creation_stack)

        if dt is not None and clock is not None:
            raise ValueError("Can only specify either a dt or a clock, not both.")

        if not isinstance(when, str):
            from brian2.core.clocks import Clock

            # Give some helpful error messages for users coming from the alpha
            # version
            if isinstance(when, Clock):
                raise TypeError(
                    "Do not use the 'when' argument for "
                    "specifying a clock, either provide a "
                    "timestep for the 'dt' argument or a Clock "
                    "object for 'clock'."
                )
            if isinstance(when, tuple):
                raise TypeError(
                    "Use the separate keyword arguments, 'dt' (or "
                    "'clock'), 'when', and 'order' instead of "
                    "providing a tuple for 'when'. Only use the "
                    "'when' argument for the scheduling slot."
                )
            # General error
            raise TypeError(
                "The 'when' argument has to be a string "
                "specifying the scheduling slot (e.g. 'start')."
            )

        Nameable.__init__(self, name)

        #: The clock used for simulating this object
        self._clock = clock
        if clock is None:
            from brian2.core.clocks import Clock, defaultclock

            if dt is not None:
                self._clock = Clock(dt=dt, name=self.name + "_clock*")
            else:
                self._clock = defaultclock

        if getattr(self._clock, "_is_proxy", False):
            from brian2.devices.device import get_device

            self._clock = get_device().defaultclock

        #: Used to remember the `Network` in which this object has been included
        #: before, to raise an error if it is included in a new `Network`
        self._network = None

        #: The ID string determining when the object should be updated in `Network.run`.
        self.when = when

        #: The order in which objects with the same clock and ``when`` should be updated
        self.order = order

        self._dependencies = set()
        self._contained_objects = []
        self._code_objects = []

        self._active = True

        #: The scope key is used to determine which objects are collected by magic
        self._scope_key = self._scope_current_key

        # Make sure that keys in the namespace are valid
        if namespace is None:
            # Do not overwrite namespace if already set (e.g. in StateMonitor)
            namespace = getattr(self, "namespace", {})
        for key in namespace:
            if key.startswith("_"):
                raise ValueError(
                    "Names starting with underscores are "
                    "reserved for internal use an cannot be "
                    "defined in the namespace argument."
                )
        #: The group-specific namespace
        self.namespace = namespace

        logger.diagnostic(
            f"Created BrianObject with name {self.name}, "
            f"clock={self._clock}, "
            f"when={self.when}, order={self.order}"
        )

    #: Global key value for ipython cell restrict magic
    _scope_current_key = 0

    #: Whether or not `MagicNetwork` is invalidated when a new `BrianObject` of this type is added
    invalidates_magic_network = True

    #: Whether or not the object should be added to a `MagicNetwork`. Note that
    #: all objects in `BrianObject.contained_objects` are automatically added
    #: when the parent object is added, therefore e.g. `NeuronGroup` should set
    #: `add_to_magic_network` to ``True``, but it should not be set for all the
    #: dependent objects such as `StateUpdater`
    add_to_magic_network = False

    def __del__(self):
        # For objects that get garbage collected, raise a warning if they have
        # never been part of a network
        if (
            getattr(self, "_network", "uninitialized") is None
            and getattr(self, "group", None) is None
        ):
            logger.warn(
                f"The object '{self.name}' is getting deleted, but was never included in a network. "
                "This probably means that you did not store the object reference in a variable, "
                "or that the variable was not used to construct the network.\n"
                "The object was created here (most recent call only):\n"
                + self._creation_stack,
                name_suffix="unused_brian_object",
            )

    def add_dependency(self, obj):
        """
        Add an object to the list of dependencies. Takes care of handling
        subgroups correctly (i.e., adds its parent object).

        Parameters
        ----------
        obj : `BrianObject`
            The object that this object depends on.
        """
        from brian2.groups.subgroup import Subgroup

        if isinstance(obj, Subgroup):
            self._dependencies.add(obj.source.id)
        else:
            self._dependencies.add(obj.id)

    def before_run(self, run_namespace):
        """
        Optional method to prepare the object before a run.

        Called by `Network.after_run` before the main simulation loop starts.
        """
        for codeobj in self._code_objects:
            codeobj.before_run()

    def after_run(self):
        """
        Optional method to do work after a run is finished.

        Called by `Network.after_run` after the main simulation loop terminated.
        """
        for codeobj in self._code_objects:
            codeobj.after_run()

    def run(self):
        for codeobj in self._code_objects:
            codeobj()

    contained_objects = property(
        fget=lambda self: self._contained_objects,
        doc="""
        The list of objects contained within the `BrianObject`.

        When a `BrianObject` is added to a `Network`, its contained objects will
        be added as well. This allows for compound objects which contain
        a mini-network structure.

        Note that this attribute cannot be set directly, you need to modify
        the underlying list, e.g. ``obj.contained_objects.extend([A, B])``.
        """,
    )

    code_objects = property(
        fget=lambda self: self._code_objects,
        doc="""
        The list of `CodeObject` contained within the `BrianObject`.

        TODO: more details.

        Note that this attribute cannot be set directly, you need to modify
        the underlying list, e.g. ``obj.code_objects.extend([A, B])``.
        """,
    )

    updaters = property(
        fget=lambda self: self._updaters,
        doc="""
        The list of `Updater` that define the runtime behaviour of this object.

        TODO: more details.

        Note that this attribute cannot be set directly, you need to modify
        the underlying list, e.g. ``obj.updaters.extend([A, B])``.
        """,
    )

    clock = property(
        fget=lambda self: self._clock,
        doc="""
        The `Clock` determining when the object should be updated.

        Note that this cannot be changed after the object is
        created.
        """,
    )

    def _set_active(self, val):
        val = bool(val)
        self._active = val
        for obj in self.contained_objects:
            obj.active = val

    active = property(
        fget=lambda self: self._active,
        fset=_set_active,
        doc="""
        Whether or not the object should be run.

        Inactive objects will not have their `update`
        method called in `Network.run`. Note that setting or
        unsetting the `active` attribute will set or unset
        it for all `contained_objects`.
        """,
    )

    def __repr__(self):
        classname = self.__class__.__name__
        description = (
            f"{classname}(clock={self._clock}, when={self.when}, "
            f"order={self.order}, name={self.name!r})"
        )
        return description

    # This is a repeat from Nameable.name, but we want to get the documentation
    # here again
    name = Nameable.name


def weakproxy_with_fallback(obj):
    """
    Attempts to create a `weakproxy` to the object, but falls back to the object if not possible.
    """
    try:
        return weakref.proxy(obj)
    except TypeError:
        return obj


def device_override(name):
    """
    Decorates a function/method to allow it to be overridden by the current `Device`.

    The ``name`` is the function name in the `Device` to use as an override if it exists.

    The returned function has an additional attribute ``original_function``
    which is a reference to the original, undecorated function.
    """

    def device_override_decorator(func):
        def device_override_decorated_function(*args, **kwds):
            from brian2.devices.device import get_device

            curdev = get_device()
            if hasattr(curdev, name):
                return getattr(curdev, name)(*args, **kwds)
            else:
                return func(*args, **kwds)

        device_override_decorated_function.original_function = func
        functools.update_wrapper(device_override_decorated_function, func)

        return device_override_decorated_function

    return device_override_decorator


class BrianObjectException(Exception):
    """
    High level exception that adds extra Brian-specific information to exceptions

    This exception should only be raised at a fairly high level in Brian code to
    pass information back to the user. It adds extra information about where a
    `BrianObject` was defined to better enable users to locate the source of
    problems.

    Parameters
    ----------

    message : str
        Additional error information to add to the original exception.
    brianobj : BrianObject
        The object that caused the error to happen.
    original_exception : Exception
        The original exception that was raised.
    """

    def __init__(self, message, brianobj):
        self._brian_message = message
        self._brian_objname = brianobj.name
        self._brian_objcreate = (
            "Object was created here (most recent call only, full details in "
            "debug log):\n" + brianobj._creation_stack
        )
        full_stack = "Object was created here:\n" + brianobj._full_creation_stack
        logger.diagnostic(
            "Error was encountered with object "
            f"'{self._brian_objname}':\n"
            f"{full_stack}"
        )

    def __str__(self):
        return (
            f"Error encountered with object named '{self._brian_objname}'.\n"
            f"{self._brian_objcreate}\n\n"
            f"{self._brian_message} "
            "(See above for original error message and traceback.)"
        )


def brian_object_exception(message, brianobj, original_exception):
    """
    Returns a `BrianObjectException` derived from the original exception.

    Creates a new class derived from the class of the original exception
    and `BrianObjectException`. This allows exception handling code to
    respond both to the original exception class and `BrianObjectException`.

    See `BrianObjectException` for arguments and notes.
    """

    raise NotImplementedError(
        "The brian_object_exception function is no longer used. "
        "Raise a BrianObjectException directly."
    )
