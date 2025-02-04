"""
This model defines the `NeuronGroup`, the core of most simulations.
"""

import numbers
import string
from collections.abc import MutableMapping, Sequence

import numpy as np
import sympy
from pyparsing import Word

from brian2.codegen.translation import analyse_identifiers
from brian2.core.preferences import prefs
from brian2.core.spikesource import SpikeSource
from brian2.core.variables import Variables
from brian2.equations.equations import (
    DIFFERENTIAL_EQUATION,
    PARAMETER,
    SUBEXPRESSION,
    Equations,
    check_subexpressions,
    extract_constant_subexpressions,
)
from brian2.equations.refractory import add_refractoriness
from brian2.parsing.expressions import (
    is_boolean_expression,
    parse_expression_dimensions,
)
from brian2.stateupdaters.base import StateUpdateMethod
from brian2.units.allunits import second
from brian2.units.fundamentalunits import (
    DIMENSIONLESS,
    Quantity,
    fail_for_dimension_mismatch,
)
from brian2.utils.logger import get_logger
from brian2.utils.stringtools import get_identifiers

from .group import CodeRunner, Group, get_dtype
from .subgroup import Subgroup

__all__ = ["NeuronGroup"]

logger = get_logger(__name__)


IDENTIFIER = Word(
    f"{string.ascii_letters}_", f"{string.ascii_letters + string.digits}_"
).setResultsName("identifier")


def _valid_event_name(event_name):
    """
    Helper function to check whether a name is a valid name for an event.

    Parameters
    ----------
    event_name : str
        The name to check

    Returns
    -------
    is_valid : bool
        Whether the given name is valid
    """
    parse_result = list(IDENTIFIER.scanString(event_name))

    # parse_result[0][0][0] refers to the matched string -- this should be the
    # full identifier, if not it is an illegal identifier like "3foo" which only
    # matched on "foo"
    return len(parse_result) == 1 and parse_result[0][0][0] == event_name


def _guess_membrane_potential(equations):
    """
    Little helper function to guess which variable represents the membrane
    potential. This follows the same logic as in Brian1 but is only used to
    give a suggestion in the error message when a Brian1-style syntax is used
    for threshold or reset.
    """
    if len(equations) == 1:
        return list(equations.keys())[0]
    for name in equations:
        if name in ["V", "v", "Vm", "vm"]:
            return name

    # nothing found
    return None


# Note that we do not register this function with
# Equations.register_identifier_check, because we do not want this check to
# apply unconditionally to all equation objects ("x_post = ... : ... (summed)"
# needs to be allowed)
def check_identifier_pre_post(identifier):
    "Do not allow names ending in ``_pre`` or ``_post`` to avoid confusion."
    if identifier.endswith("_pre") or identifier.endswith("_post"):
        raise ValueError(
            f"'{identifier}' cannot be used as a variable name, the "
            "'_pre' and '_post' suffixes are used to refer to pre- and "
            "post-synaptic variables in synapses."
        )


def to_start_stop(item, N):
    """
    Helper function to transform a single number, a slice or an array of
    contiguous indices to a start and stop value. This is used to allow for
    some flexibility in the syntax of specifying subgroups in `.NeuronGroup`
    and `.SpatialNeuron`.

    Parameters
    ----------
    item : slice, int or sequence
        The slice, index, or sequence of indices to use. Note that a sequence
        of indices has to be a sorted ascending sequence of subsequent integers.
    N : int
        The total number of elements in the group.

    Returns
    -------
    start : int
        The start value of the slice.
    stop : int
        The stop value of the slice.

    Examples
    --------
    >>> from brian2.groups.neurongroup import to_start_stop
    >>> to_start_stop(slice(3, 6), 10)
    (3, 6)
    >>> to_start_stop(slice(3, None), 10)
    (3, 10)
    >>> to_start_stop(5, 10)
    (5, 6)
    >>> to_start_stop([3, 4, 5], 10)
    (3, 6)
    >>> to_start_stop([3, 5, 7], 10)
    Traceback (most recent call last):
        ...
    IndexError: Subgroups can only be constructed using contiguous indices.

    """
    if isinstance(item, slice):
        start, stop, step = item.indices(N)
    elif isinstance(item, numbers.Integral):
        start = item
        stop = item + 1
        step = 1
    elif isinstance(item, (Sequence, np.ndarray)) and not isinstance(item, str):
        if not (len(item) > 0 and np.all(np.diff(item) == 1)):
            raise IndexError(
                "Subgroups can only be constructed using contiguous indices."
            )
        if not np.issubdtype(np.asarray(item).dtype, np.integer):
            raise TypeError("Subgroups can only be constructed using integer values.")
        start = int(item[0])
        stop = int(item[-1]) + 1
        step = 1
    else:
        raise TypeError(
            "Subgroups can only be constructed using slicing "
            "syntax, a single index, or an array of contiguous "
            "indices."
        )
    if step != 1:
        raise IndexError("Subgroups have to be contiguous")
    if start >= stop:
        raise IndexError(
            f"Illegal start/end values for subgroup, {int(start)}>={int(stop)}"
        )
    if start >= N:
        raise IndexError(f"Illegal start value for subgroup, {int(start)}>={int(N)}")
    if stop > N:
        raise IndexError(f"Illegal stop value for subgroup, {int(stop)}>{int(N)}")
    if start < 0:
        raise IndexError("Indices have to be positive.")
    return start, stop


class StateUpdater(CodeRunner):
    """
    The `CodeRunner` that updates the state variables of a `NeuronGroup`
    at every timestep.
    """

    def __init__(self, group, method, method_options=None):
        self.method_choice = method
        self.method_options = method_options
        CodeRunner.__init__(
            self,
            group,
            "stateupdate",
            code="",  # will be set in update_abstract_code
            clock=group.clock,
            when="groups",
            order=group.order,
            name=f"{group.name}_stateupdater",
            check_units=False,
            generate_empty_code=False,
        )

    def _get_refractory_code(self, run_namespace):
        ref = self.group._refractory
        if ref is False:
            # No refractoriness
            abstract_code = ""
        elif isinstance(ref, Quantity):
            fail_for_dimension_mismatch(
                ref,
                second,
                "Refractory period has to "
                "be specified in units "
                "of seconds but got "
                "{value}",
                value=ref,
            )
            ref = float(ref)
            if prefs.legacy.refractory_timing:
                abstract_code = f"not_refractory = (t - lastspike) > {ref}\n"
            else:
                abstract_code = f"not_refractory = timestep(t - lastspike, dt) >= timestep({ref}, dt)\n"
        else:
            identifiers = get_identifiers(ref)
            variables = self.group.resolve_all(
                identifiers, run_namespace, user_identifiers=identifiers
            )
            dims = parse_expression_dimensions(str(ref), variables)
            if dims is second.dim:
                if prefs.legacy.refractory_timing:
                    abstract_code = f"(t - lastspike) > {ref}\n"
                else:
                    abstract_code = (
                        "not_refractory = timestep(t - lastspike, dt) >="
                        f" timestep({ref}, dt)\n"
                    )
            elif dims is DIMENSIONLESS:
                if not is_boolean_expression(str(ref), variables):
                    raise TypeError(
                        "Refractory expression is dimensionless "
                        "but not a boolean value. It needs to "
                        "either evaluate to a timespan or to a "
                        "boolean value."
                    )
                # boolean condition
                # we have to be a bit careful here, we can't just use the given
                # condition as it is, because we only want to *leave*
                # refractoriness, based on the condition
                abstract_code = f"not_refractory = not_refractory or not ({ref})\n"
            else:
                raise TypeError(
                    "Refractory expression has to evaluate to a "
                    "timespan or a boolean value, expression"
                    f"'{ref}' has units {dims} instead"
                )
        return abstract_code

    def update_abstract_code(self, run_namespace):
        # Update the not_refractory variable for the refractory period mechanism
        self.abstract_code = self._get_refractory_code(run_namespace=run_namespace)

        # Get the names used in the refractory code
        _, used_known, unknown = analyse_identifiers(
            self.abstract_code, self.group.variables, recursive=True
        )

        # Get all names used in the equations (and always get "dt")
        names = self.group.equations.names
        external_names = self.group.equations.identifiers | {"dt"}

        variables = self.group.resolve_all(
            used_known | unknown | names | external_names,
            run_namespace,
            # we don't need to raise any warnings
            # for the user here, warnings will
            # be raised in create_runner_codeobj
            user_identifiers=set(),
        )
        if len(self.group.equations.diff_eq_names) > 0:
            stateupdate_output = StateUpdateMethod.apply_stateupdater(
                self.group.equations,
                variables,
                self.method_choice,
                method_options=self.method_options,
                group_name=self.group.name,
            )
            if isinstance(stateupdate_output, str):
                self.abstract_code += stateupdate_output
            else:
                # Note that the reason to send self along with this method is so the StateUpdater
                # can be modified! i.e. in GSL StateUpdateMethod a custom CodeObject gets added
                # to the StateUpdater together with some auxiliary information
                self.abstract_code += stateupdate_output(self)

        user_code = "\n".join(
            [
                f"{var} = {expr}"
                for var, expr in self.group.equations.get_substituted_expressions(
                    variables
                )
            ]
        )
        self.user_code = user_code


class SubexpressionUpdater(CodeRunner):
    """
    The `CodeRunner` that updates the state variables storing the values of
    subexpressions that have been marked as "constant over dt".
    """

    def __init__(self, group, subexpressions, when="before_start"):
        code_lines = []
        for subexpr in subexpressions.ordered:
            code_lines.append(f"{subexpr.varname} = {subexpr.expr}")
        code = "\n".join(code_lines)
        CodeRunner.__init__(
            self,
            group,
            "stateupdate",
            code=code,  # will be set in update_abstract_code
            clock=group.clock,
            when=when,
            order=group.order,
            name=f"{group.name}_subexpression_update*",
        )


class Thresholder(CodeRunner):
    """
    The `CodeRunner` that applies the threshold condition to the state
    variables of a `NeuronGroup` at every timestep and sets its ``spikes``
    and ``refractory_until`` attributes.
    """

    def __init__(self, group, when="thresholds", event="spike"):
        self.event = event
        if group._refractory is False or event != "spike":
            template_kwds = {"_uses_refractory": False}
            needed_variables = []
        else:
            template_kwds = {"_uses_refractory": True}
            needed_variables = ["t", "not_refractory", "lastspike"]
        # Since this now works for general events not only spikes, we have to
        # pass the information about which variable to use to the template,
        # it can not longer simply refer to "_spikespace"
        eventspace_name = f"_{event}space"
        template_kwds["eventspace_variable"] = group.variables[eventspace_name]
        needed_variables.append(eventspace_name)
        self.variables = Variables(self)
        self.variables.add_auxiliary_variable("_cond", dtype=bool)
        CodeRunner.__init__(
            self,
            group,
            "threshold",
            code="",  # will be set in update_abstract_code
            clock=group.clock,
            when=when,
            order=group.order,
            name=f"{group.name}_{event}_thresholder",
            needed_variables=needed_variables,
            template_kwds=template_kwds,
        )

    def update_abstract_code(self, run_namespace):
        code = self.group.events[self.event]
        # Raise a useful error message when the user used a Brian1 syntax
        if not isinstance(code, str):
            if isinstance(code, Quantity):
                t = "a quantity"
            else:
                t = f"{type(code)}"
            error_msg = f"Threshold condition has to be a string, not {t}."
            if self.event == "spike":
                try:
                    vm_var = _guess_membrane_potential(self.group.equations)
                except AttributeError:  # not a group with equations...
                    vm_var = None
                if vm_var is not None:
                    error_msg += f" Probably you intended to use '{vm_var} > ...'?"
            raise TypeError(error_msg)

        self.user_code = f"_cond = {code}"

        identifiers = get_identifiers(code)
        variables = self.group.resolve_all(
            identifiers, run_namespace, user_identifiers=identifiers
        )
        if not is_boolean_expression(code, variables):
            raise TypeError(f"Threshold condition '{code}' is not a boolean expression")
        if self.group._refractory is False or self.event != "spike":
            self.abstract_code = f"_cond = {code}"
        else:
            self.abstract_code = f"_cond = ({code}) and not_refractory"


class Resetter(CodeRunner):
    """
    The `CodeRunner` that applies the reset statement(s) to the state
    variables of neurons that have spiked in this timestep.
    """

    def __init__(self, group, when="resets", order=None, event="spike"):
        self.event = event
        # Since this now works for general events not only spikes, we have to
        # pass the information about which variable to use to the template,
        # it can not longer simply refer to "_spikespace"
        eventspace_name = f"_{event}space"
        template_kwds = {"eventspace_variable": group.variables[eventspace_name]}
        needed_variables = [eventspace_name]
        order = order if order is not None else group.order
        CodeRunner.__init__(
            self,
            group,
            "reset",
            code="",  # will be set in update_abstract_code
            clock=group.clock,
            when=when,
            order=order,
            name=f"{group.name}_{event}_resetter",
            override_conditional_write=["not_refractory"],
            needed_variables=needed_variables,
            template_kwds=template_kwds,
        )

    def update_abstract_code(self, run_namespace):
        code = self.group.event_codes[self.event]
        # Raise a useful error message when the user used a Brian1 syntax
        if not isinstance(code, str):
            if isinstance(code, Quantity):
                t = "a quantity"
            else:
                t = f"{type(code)}"
            error_msg = f"Reset statement has to be a string, not {t}."
            if self.event == "spike":
                vm_var = _guess_membrane_potential(self.group.equations)
                if vm_var is not None:
                    error_msg += f" Probably you intended to use '{vm_var} = ...'?"
            raise TypeError(error_msg)

        self.abstract_code = code


class NeuronGroup(Group, SpikeSource):
    """
    A group of neurons.


    Parameters
    ----------
    N : int
        Number of neurons in the group.
    model : str, `Equations`
        The differential equations defining the group
    method : (str, function), optional
        The numerical integration method. Either a string with the name of a
        registered method (e.g. "euler") or a function that receives an
        `Equations` object and returns the corresponding abstract code. If no
        method is specified, a suitable method will be chosen automatically.
    threshold : str, optional
        The condition which produces spikes. Should be a single line boolean
        expression.
    reset : str, optional
        The (possibly multi-line) string with the code to execute on reset.
    refractory : {str, `Quantity`}, optional
        Either the length of the refractory period (e.g. ``2*ms``), a string
        expression that evaluates to the length of the refractory period
        after each spike (e.g. ``'(1 + rand())*ms'``), or a string expression
        evaluating to a boolean value, given the condition under which the
        neuron stays refractory after a spike (e.g. ``'v > -20*mV'``)
    events : dict, optional
        User-defined events in addition to the "spike" event defined by the
        ``threshold``. Has to be a mapping of strings (the event name) to
        strings (the condition) that will be checked.
    namespace: dict, optional
        A dictionary mapping identifier names to objects. If not given, the
        namespace will be filled in at the time of the call of `Network.run`,
        with either the values from the ``namespace`` argument of the
        `Network.run` method or from the local context, if no such argument is
        given.
    dtype : (`dtype`, `dict`), optional
        The `numpy.dtype` that will be used to store the values, or a
        dictionary specifying the type for variable names. If a value is not
        provided for a variable (or no value is provided at all), the preference
        setting `core.default_float_dtype` is used.
    codeobj_class : class, optional
        The `CodeObject` class to run code with.
    dt : `Quantity`, optional
        The time step to be used for the simulation. Cannot be combined with
        the `clock` argument.
    clock : `Clock`, optional
        The update clock to be used. If neither a clock, nor the `dt` argument
        is specified, the `defaultclock` will be used.
    order : int, optional
        The priority of of this group for operations occurring at the same time
        step and in the same scheduling slot. Defaults to 0.
    name : str, optional
        A unique name for the group, otherwise use ``neurongroup_0``, etc.

    Notes
    -----
    `NeuronGroup` contains a `StateUpdater`, `Thresholder` and `Resetter`, and
    these are run at the 'groups', 'thresholds' and 'resets' slots (i.e. the
    values of their `when` attribute take these values). The `order`
    attribute will be passed down to the contained objects but can be set
    individually by setting the `order` attribute of the `state_updater`,
    `thresholder` and `resetter` attributes, respectively.
    """

    add_to_magic_network = True

    def __init__(
        self,
        N,
        model,
        method=("exact", "euler", "heun"),
        method_options=None,
        threshold=None,
        reset=None,
        refractory=False,
        events=None,
        namespace=None,
        dtype=None,
        dt=None,
        clock=None,
        order=0,
        name="neurongroup*",
        codeobj_class=None,
    ):
        Group.__init__(
            self,
            dt=dt,
            clock=clock,
            when="start",
            order=order,
            namespace=namespace,
            name=name,
        )
        if dtype is None:
            dtype = {}
        if isinstance(dtype, MutableMapping):
            dtype["lastspike"] = self._clock.variables["t"].dtype

        self.codeobj_class = codeobj_class

        try:
            self._N = N = int(N)
        except ValueError:
            if isinstance(N, str):
                raise TypeError(
                    "First NeuronGroup argument should be size, not equations."
                )
            raise
        if N < 1:
            raise ValueError(f"NeuronGroup size should be at least 1, was {str(N)}")

        self.start = 0
        self.stop = self._N

        ##### Prepare and validate equations
        if isinstance(model, str):
            model = Equations(model)
        if not isinstance(model, Equations):
            raise TypeError(
                "model has to be a string or an Equations "
                f"object, is '{type(model)}' instead."
            )

        # Check flags
        model.check_flags(
            {
                DIFFERENTIAL_EQUATION: ("unless refractory",),
                PARAMETER: ("constant", "shared", "linked"),
                SUBEXPRESSION: ("shared", "constant over dt"),
            }
        )

        # add refractoriness
        #: The original equations as specified by the user (i.e. without
        #: the multiplied `int(not_refractory)` term for equations marked as
        #: `(unless refractory)`)
        self.user_equations = model
        if refractory is not False:
            model = add_refractoriness(model)
        uses_refractoriness = len(model) and any(
            [
                "unless refractory" in eq.flags
                for eq in model.values()
                if eq.type == DIFFERENTIAL_EQUATION
            ]
        )

        # Separate subexpressions depending whether they are considered to be
        # constant over a time step or not
        model, constant_over_dt = extract_constant_subexpressions(model)
        self.equations = model

        self._linked_variables = set()
        logger.diagnostic(
            f"Creating NeuronGroup of size {self._N}, equations {self.equations}."
        )

        # All of the following will be created in before_run

        #: The refractory condition or timespan
        self._refractory = refractory
        if uses_refractoriness and refractory is False:
            logger.warn(
                'Model equations use the "unless refractory" flag but '
                "no refractory keyword was given.",
                "no_refractory",
            )

        #: The state update method selected by the user
        self.method_choice = method

        if events is None:
            events = {}

        if threshold is not None and (reset is None and refractory is False):
            if not ("rand(" in threshold or "randn(" in threshold):
                logger.warn(
                    f"The NeuronGroup '{self.name}' sets a threshold but "
                    "neither a reset condition nor a refractory "
                    "condition has been set. Did you forget either of "
                    "those? If this was intended, set the reset "
                    "argument to an empty string in order to avoid "
                    "this warning.",
                    name_suffix="only_threshold",
                )

        if threshold is not None:
            if "spike" in events:
                raise ValueError(
                    "The NeuronGroup defines both a threshold and a 'spike' event"
                )
            events["spike"] = threshold

        # Setup variables
        # Since we have to create _spikespace and possibly other "eventspace"
        # variables, we pass the supported events
        self._create_variables(dtype, events=list(events.keys()))

        #: Events supported by this group
        self.events = events

        #: Code that is triggered on events (e.g. reset)
        self.event_codes = {}

        #: Checks the spike threshold (or abitrary user-defined events)
        self.thresholder = {}

        #: Reset neurons which have spiked (or perform arbitrary actions for
        #: user-defined events)
        self.resetter = {}

        for event_name in events.keys():
            if not isinstance(event_name, str):
                raise TypeError(
                    "Keys in the 'events' dictionary have to be "
                    f"strings, not type {event_name}."
                )
            if not _valid_event_name(event_name):
                raise TypeError(
                    f"The name '{event_name}' cannot be used as an event name."
                )
            # By default, user-defined events are checked after the threshold
            when = "thresholds" if event_name == "spike" else "after_thresholds"
            # creating a Thresholder will take care of checking the validity
            # of the condition
            thresholder = Thresholder(self, event=event_name, when=when)
            self.thresholder[event_name] = thresholder
            self.contained_objects.append(thresholder)

        if reset is not None:
            self.run_on_event("spike", reset, when="resets")

        #: Performs numerical integration step
        self.state_updater = StateUpdater(self, method, method_options)
        self.contained_objects.append(self.state_updater)

        #: Update the "constant over a time step" subexpressions
        self.subexpression_updater = None
        if len(constant_over_dt):
            self.subexpression_updater = SubexpressionUpdater(self, constant_over_dt)
            self.contained_objects.append(self.subexpression_updater)

        if refractory is not False:
            # Set the refractoriness information
            self.variables["lastspike"].set_value(-1e4 * second)
            self.variables["not_refractory"].set_value(True)

        # Activate name attribute access
        self._enable_group_attributes()

    @property
    def spikes(self):
        """
        The spikes returned by the most recent thresholding operation.
        """
        # Note that we have to directly access the ArrayVariable object here
        # instead of using the Group mechanism by accessing self._spikespace
        # Using the latter would cut _spikespace to the length of the group
        spikespace = self.variables["_spikespace"].get_value()
        return spikespace[: spikespace[-1]]

    def state(self, name, use_units=True, level=0):
        try:
            return Group.state(self, name, use_units=use_units, level=level + 1)
        except KeyError as ex:
            if name in self._linked_variables:
                raise TypeError(f"Link target for variable {name} has not been set.")
            else:
                raise ex

    def run_on_event(self, event, code, when="after_resets", order=None):
        """
        Run code triggered by a custom-defined event (see `NeuronGroup`
        documentation for the specification of events).The created `Resetter`
        object will be automatically added to the group, it therefore does not
        need to be added to the network manually. However, a reference to the
        object will be returned, which can be used to later remove it from the
        group or to set it to inactive.

        Parameters
        ----------
        event : str
            The name of the event that should trigger the code
        code : str
            The code that should be executed
        when : str, optional
            The scheduling slot that should be used to execute the code.
            Defaults to `'after_resets'`. See :ref:`scheduling` for possible values.
        order : int, optional
            The order for operations in the same scheduling slot. Defaults to
            the order of the `NeuronGroup`.

        Returns
        -------
        obj : `Resetter`
            A reference to the object that will be run.
        """
        if event not in self.events:
            error_message = f"Unknown event '{event}'."
            if event == "spike":
                error_message += " Did you forget to define a threshold?"
            raise ValueError(error_message)
        if event in self.resetter:
            raise ValueError(
                "Cannot add code for event '%s', code for this "
                "event has already been added." % event
            )
        self.event_codes[event] = code
        resetter = Resetter(self, when=when, order=order, event=event)
        self.resetter[event] = resetter
        self.contained_objects.append(resetter)

        return resetter

    def set_event_schedule(self, event, when="after_thresholds", order=None):
        """
        Change the scheduling slot for checking the condition of an event.

        Parameters
        ----------
        event : str
            The name of the event for which the scheduling should be changed
        when : str, optional
            The scheduling slot that should be used to check the condition.
            Defaults to `'after_thresholds'`. See :ref:`scheduling` for possible values.
        order : int, optional
            The order for operations in the same scheduling slot. Defaults to
            the order of the `NeuronGroup`.
        """
        if event not in self.thresholder:
            raise ValueError(f"Unknown event '{event}'.")
        order = order if order is not None else self.order
        self.thresholder[event].when = when
        self.thresholder[event].order = order

    def __getitem__(self, item):
        start, stop = to_start_stop(item, self._N)

        return Subgroup(self, start, stop)

    def _create_variables(self, user_dtype, events):
        """
        Create the variables dictionary for this `NeuronGroup`, containing
        entries for the equation variables and some standard entries.
        """
        self.variables = Variables(self)
        self.variables.add_constant("N", self._N)

        # Standard variables always present
        for event in events:
            self.variables.add_array(
                f"_{event}space", size=self._N + 1, dtype=np.int32, constant=False
            )
        # Add the special variable "i" which can be used to refer to the neuron index
        self.variables.add_arange("i", size=self._N, constant=True, read_only=True)
        # Add the clock variables
        self.variables.create_clock_variables(self._clock)

        for eq in self.equations.values():
            dtype = get_dtype(eq, user_dtype)
            check_identifier_pre_post(eq.varname)
            if eq.type in (DIFFERENTIAL_EQUATION, PARAMETER):
                if "linked" in eq.flags:
                    # 'linked' cannot be combined with other flags
                    if not len(eq.flags) == 1:
                        raise SyntaxError(
                            "The 'linked' flag cannot be combined with other flags"
                        )
                    self._linked_variables.add(eq.varname)
                else:
                    constant = "constant" in eq.flags
                    shared = "shared" in eq.flags
                    size = 1 if shared else self._N
                    self.variables.add_array(
                        eq.varname,
                        size=size,
                        dimensions=eq.dim,
                        dtype=dtype,
                        constant=constant,
                        scalar=shared,
                    )
            elif eq.type == SUBEXPRESSION:
                self.variables.add_subexpression(
                    eq.varname,
                    dimensions=eq.dim,
                    expr=str(eq.expr),
                    dtype=dtype,
                    scalar="shared" in eq.flags,
                )
            else:
                raise AssertionError(f"Unknown type of equation: {eq.eq_type}")

        # Add the conditional-write attribute for variables with the
        # "unless refractory" flag
        if self._refractory is not False:
            for eq in self.equations.values():
                if eq.type == DIFFERENTIAL_EQUATION and "unless refractory" in eq.flags:
                    not_refractory_var = self.variables["not_refractory"]
                    var = self.variables[eq.varname]
                    var.set_conditional_write(not_refractory_var)

        # Stochastic variables
        for xi in self.equations.stochastic_variables:
            self.variables.add_auxiliary_variable(xi, dimensions=(second**-0.5).dim)

        # Check scalar subexpressions
        for eq in self.equations.values():
            if eq.type == SUBEXPRESSION and "shared" in eq.flags:
                var = self.variables[eq.varname]
                for identifier in var.identifiers:
                    if identifier in self.variables:
                        if not self.variables[identifier].scalar:
                            raise SyntaxError(
                                f"Shared subexpression '{eq.varname}' "
                                "refers to non-shared variable "
                                f"'{identifier}'."
                            )

    def before_run(self, run_namespace=None):
        # Check units
        self.equations.check_units(self, run_namespace=run_namespace)
        # Check that subexpressions that refer to stateful functions are labeled
        # as "constant over dt"
        check_subexpressions(self, self.equations, run_namespace)
        super().before_run(run_namespace=run_namespace)

    def _repr_html_(self):
        text = [rf"NeuronGroup '{self.name}' with {self._N} neurons.<br>"]
        text.append(r"<b>Model:</b><nr>")
        text.append(sympy.latex(self.equations))

        def add_event_to_text(event):
            if event == "spike":
                event_header = "Spiking behaviour"
                event_condition = "Threshold condition"
                event_code = "Reset statement(s)"
            else:
                event_header = f'Event "{event}"'
                event_condition = "Event condition"
                event_code = "Executed statement(s)"
            condition = self.events[event]
            text.append(
                rf'<b>{event_header}:</b><ul style="list-style-type: none; margin-top:'
                r' 0px;">'
            )
            text.append(rf"<li><i>{event_condition}: </i>")
            text.append(f"<code>{str(condition)}</code></li>")
            statements = self.event_codes.get(event, None)
            if statements is not None:
                text.append(rf"<li><i>{event_code}:</i>")
                if "\n" in str(statements):
                    text.append("</br>")
                text.append(rf"<code>{str(statements)}</code></li>")
            text.append("</ul>")

        if "spike" in self.events:
            add_event_to_text("spike")
        for event in self.events:
            if event != "spike":
                add_event_to_text(event)

        return "\n".join(text)
