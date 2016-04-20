'''
Classes used to specify the type of a function, variable or common
sub-expression.
'''
import collections
import functools
import numbers

import sympy
import numpy as np

from brian2.utils.stringtools import get_identifiers, word_substitute
from brian2.units.fundamentalunits import (Quantity, Unit, DIMENSIONLESS,
                                           fail_for_dimension_mismatch,
                                           have_same_dimensions, get_unit)
from brian2.utils.logger import get_logger

from .base import weakproxy_with_fallback, device_override
from .preferences import prefs

__all__ = ['Variable',
           'Constant',
           'ArrayVariable',
           'DynamicArrayVariable',
           'Subexpression',
           'AuxiliaryVariable',
           'VariableView',
           'Variables',
           'LinkedVariable',
           'linked_var'
           ]


logger = get_logger(__name__)


def get_dtype(obj):
    '''
    Helper function to return the `numpy.dtype` of an arbitrary object.

    Parameters
    ----------
    obj : object
        Any object (but typically some kind of number or array).

    Returns
    -------
    dtype : `numpy.dtype`
        The type of the given object.
    '''
    if hasattr(obj, 'dtype'):
        return obj.dtype
    else:
        return np.obj2sctype(type(obj))


def get_dtype_str(val):
    '''
    Returns canonical string representation of the dtype of a value or dtype
    
    Returns
    -------
    
    dtype_str : str
        The numpy dtype name
    '''
    if isinstance(val, np.dtype):
        return val.name
    if isinstance(val, type):
        return get_dtype_str(val())

    is_bool = (val is True or
               val is False or
               val is np.True_ or
               val is np.False_)
    if is_bool:
        return 'bool'
    if hasattr(val, 'dtype'):
        return get_dtype_str(val.dtype)
    if isinstance(val, numbers.Number):
        return get_dtype_str(np.array(val).dtype)
    
    return 'unknown[%s, %s]' % (str(val), val.__class__.__name__)


def variables_by_owner(variables, owner):
    owner_name = getattr(owner, 'name', None)
    return dict([(varname, var) for varname, var in variables.iteritems()
                 if getattr(var.owner, 'name', None) is owner_name])


class Variable(object):
    '''
    An object providing information about model variables (including implicit
    variables such as ``t`` or ``xi``). This class should never be
    instantiated outside of testing code, use one of its subclasses instead.
    
    Parameters
    ----------
    name : 'str'
        The name of the variable. Note that this refers to the *original*
        name in the owning group. The same variable may be known under other
        names in other groups (e.g. the variable ``v`` of a `NeuronGroup` is
        known as ``v_post`` in a `Synapse` connecting to the group).
    unit : `Unit`
        The unit of the variable.
    owner : `Nameable`, optional
        The object that "owns" this variable, e.g. the `NeuronGroup` or
        `Synapses` object that declares the variable in its model equations.
        Defaults to ``None`` (the value used for `Variable` objects without an
        owner, e.g. external `Constant`\ s).
    dtype : `dtype`, optional
        The dtype used for storing the variable. Defaults to the preference
        `core.default_scalar.dtype`.
    scalar : bool, optional
        Whether the variable is a scalar value (``True``) or vector-valued, e.g.
        defined for every neuron (``False``). Defaults to ``False``.
    constant: bool, optional
        Whether the value of this variable can change during a run. Defaults
        to ``False``.
    read_only : bool, optional
        Whether this is a read-only variable, i.e. a variable that is set
        internally and cannot be changed by the user (this is used for example
        for the variable ``N``, the number of neurons in a group). Defaults
        to ``False``.
    array : bool, optional
        Whether this variable is an array. Allows for simpler check than testing
        ``isinstance(var, ArrayVariable)``. Defaults to ``False``.
    '''
    def __init__(self, name, unit, owner=None, dtype=None, scalar=False,
                 constant=False, read_only=False, dynamic=False, array=False):
        if not isinstance(unit, Unit):
            if isinstance(unit, Quantity):
                unit = get_unit(unit)
            elif unit == 1:
                unit = Unit(1)
            else:
                raise TypeError(('unit argument has to be a Unit object, was '
                                 'type %s instead') % type(unit))
        #: The variable's unit.
        self.unit = unit

        #: The variable's name.
        self.name = name

        #: The `Group` to which this variable belongs.
        self.owner = weakproxy_with_fallback(owner) if owner is not None else None

        #: The dtype used for storing the variable.
        self.dtype = dtype
        if dtype is None:
            self.dtype = prefs.core.default_float_dtype

        if self.is_boolean:
            if not have_same_dimensions(unit, 1):
                raise ValueError('Boolean variables can only be dimensionless')

        #: Whether the variable is a scalar
        self.scalar = scalar

        #: Whether the variable is constant during a run
        self.constant = constant

        #: Whether the variable is read-only
        self.read_only = read_only

        #: Whether the variable is dynamically sized (only for non-scalars)
        self.dynamic = dynamic

        #: Whether the variable is an array
        self.array = array

    @property
    def is_boolean(self):
        return np.issubdtype(np.bool, self.dtype)

    @property
    def dim(self):
        '''
        The dimensions of this variable.
        '''
        return self.unit.dim
    
    @property
    def dtype_str(self):
        '''
        String representation of the numpy dtype
        '''
        return get_dtype_str(self)

    def get_value(self):
        '''
        Return the value associated with the variable (without units). This
        is the way variables are accessed in generated code.
        '''
        raise TypeError('Cannot get value for variable %s' % self)

    def set_value(self, value):
        '''
        Set the value associated with the variable.
        '''
        raise TypeError('Cannot set value for variable %s' % self)

    def get_value_with_unit(self):
        '''
        Return the value associated with the variable (with units).
        '''
        return Quantity(self.get_value(), self.unit.dimensions)

    def get_addressable_value(self, name, group):
        '''
        Get the value (without units) of this variable in a form that can be
        indexed in the context of a group. For example, if a
        postsynaptic variable ``x`` is accessed in a synapse ``S`` as
        ``S.x_post``, the synaptic indexing scheme can be used.

        Parameters
        ----------
        name : str
            The name of the variable
        group : `Group`
            The group providing the context for the indexing. Note that this
            `group` is not necessarily the same as `Variable.owner`: a variable
            owned by a `NeuronGroup` can be indexed in a different way if
            accessed via a `Synapses` object.

        Returns
        -------
        variable : object
            The variable in an indexable form (without units).
        '''
        return self.get_value()

    def get_addressable_value_with_unit(self, name, group):
        '''
        Get the value (with units) of this variable in a form that can be
        indexed in the context of a group. For example, if a postsynaptic
        variable ``x`` is accessed in a synapse ``S`` as ``S.x_post``, the
        synaptic indexing scheme can be used.

        Parameters
        ----------
        name : str
            The name of the variable
        group : `Group`
            The group providing the context for the indexing. Note that this
            `group` is not necessarily the same as `Variable.owner`: a variable
            owned by a `NeuronGroup` can be indexed in a different way if
            accessed via a `Synapses` object.

        Returns
        -------
        variable : object
            The variable in an indexable form (with units).
        '''
        return self.get_value_with_unit()

    def get_len(self):
        '''
        Get the length of the value associated with the variable or ``0`` for
        a scalar variable.
        '''
        if self.scalar:
            return 0
        else:
            return len(self.get_value())

    def __len__(self):
        return self.get_len()

    def __repr__(self):
        description = ('<{classname}(unit={unit}, '
                       ' dtype={dtype}, scalar={scalar}, constant={constant},'
                       ' read_only={read_only})>')
        return description.format(classname=self.__class__.__name__,
                                  unit=repr(self.unit),
                                  dtype=repr(self.dtype),
                                  scalar=repr(self.scalar),
                                  constant=repr(self.constant),
                                  read_only=repr(self.read_only))


# ------------------------------------------------------------------------------
# Concrete classes derived from `Variable` -- these are the only ones ever
# instantiated.
# ------------------------------------------------------------------------------

class Constant(Variable):
    '''
    A scalar constant (e.g. the number of neurons ``N``). Information such as
    the dtype or whether this variable is a boolean are directly derived from
    the `value`. Most of the time `Variables.add_constant` should be used
    instead of instantiating this class directly.

    Parameters
    ----------
    name : str
        The name of the variable
    unit : `Unit`
        The unit of the variable. Note that the variable itself (as referenced
        by value) should never have units attached.
    value: reference to the variable value
        The value of the constant.
    owner : `Nameable`, optional
        The object that "owns" this variable, for constants that belong to a
        specific group, e.g. the ``N`` constant for a `NeuronGroup`. External
        constants will have ``None`` (the default value).
    '''
    def __init__(self, name, unit, value, owner=None):
        # Determine the type of the value
        is_bool = (value is True or
                   value is False or
                   value is np.True_ or
                   value is np.False_)

        if is_bool:
            dtype = np.bool
        else:
            dtype = get_dtype(value)

        # Use standard Python types if possible for numpy scalars (generates
        # nicer code for C++ when using weave)
        if getattr(value, 'shape', None) == () and hasattr(value, 'dtype'):
            numpy_type = value.dtype
            if np.can_cast(numpy_type, np.int_):
                value = int(value)
            elif np.can_cast(numpy_type, np.float_):
                value = float(value)
            elif np.can_cast(numpy_type, np.complex_):
                value = complex(value)
            elif value is np.True_:
                value = True
            elif value is np.False_:
                value = False

        #: The constant's value
        self.value = value

        super(Constant, self).__init__(unit=unit, name=name, owner=owner,
                                       dtype=dtype, scalar=True, constant=True,
                                       read_only=True)

    def get_value(self):
        return self.value


class AuxiliaryVariable(Variable):
    '''
    Variable description for an auxiliary variable (most likely one that is
    added automatically to abstract code, e.g. ``_cond`` for a threshold
    condition), specifying its type and unit for code generation. Most of the
    time `Variables.add_auxiliary_variable` should be used instead of
    instantiating this class directly.

    Parameters
    ----------
    name : str
        The name of the variable
    unit : `Unit`
        The unit of the variable.
    dtype : `dtype`, optional
        The dtype used for storing the variable. If none is given, defaults
        to `core.default_float_dtype`.
    scalar : bool, optional
        Whether the variable is a scalar value (``True``) or vector-valued, e.g.
        defined for every neuron (``False``). Defaults to ``False``.
    '''
    def __init__(self, name, unit, dtype=None, scalar=False):
        super(AuxiliaryVariable, self).__init__(unit=unit,
                                                name=name, dtype=dtype,
                                                scalar=scalar)

    def get_value(self):
        raise TypeError('Cannot get the value for an auxiliary variable.')


class ArrayVariable(Variable):
    '''
    An object providing information about a model variable stored in an array
    (for example, all state variables). Most of the time `Variables.add_array`
    should be used instead of instantiating this class directly.

    Parameters
    ----------
    name : 'str'
        The name of the variable. Note that this refers to the *original*
        name in the owning group. The same variable may be known under other
        names in other groups (e.g. the variable ``v`` of a `NeuronGroup` is
        known as ``v_post`` in a `Synapse` connecting to the group).
    unit : `Unit`
        The unit of the variable
    owner : `Nameable`
        The object that "owns" this variable, e.g. the `NeuronGroup` or
        `Synapses` object that declares the variable in its model equations.
    size : int
        The size of the array
    device : `Device`
        The device responsible for the memory access.
    dtype : `dtype`, optional
        The dtype used for storing the variable. If none is given, defaults
        to `core.default_float_dtype`.
    constant : bool, optional
        Whether the variable's value is constant during a run.
        Defaults to ``False``.
    scalar : bool, optional
        Whether this array is a 1-element array that should be treated like a
        scalar (e.g. for a single delay value across synapses). Defaults to
        ``False``.
    read_only : bool, optional
        Whether this is a read-only variable, i.e. a variable that is set
        internally and cannot be changed by the user. Defaults
        to ``False``.
    unique : bool, optional
        Whether the values in this array are all unique. This information is
        only important for variables used as indices and does not have to
        reflect the actual contents of the array but only the possibility of
        non-uniqueness (e.g. synaptic indices are always unique but the
        corresponding pre- and post-synaptic indices are not). Defaults to
        ``False``.
    '''
    def __init__(self, name, unit, owner, size, device, dtype=None,
                 constant=False, scalar=False, read_only=False, dynamic=False,
                 unique=False):
        super(ArrayVariable, self).__init__(unit=unit, name=name, owner=owner,
                                            dtype=dtype, scalar=scalar,
                                            constant=constant,
                                            read_only=read_only,
                                            dynamic=dynamic,
                                            array=True)

        #: Wether all values in this arrays are necessarily unique (only
        #: relevant for index variables).
        self.unique = unique

        #: The `Device` responsible for memory access.
        self.device = device

        #: The size of this variable.
        self.size = size

        if scalar and size != 1:
            raise ValueError(('Scalar variables need to have size 1, not '
                              'size %d.') % size)

        #: Another variable, on which the write is conditioned (e.g. a variable
        #: denoting the absence of refractoriness)
        self.conditional_write = None

    def set_conditional_write(self, var):
        if not var.is_boolean:
            raise TypeError(('A variable can only be conditionally writeable '
                             'depending on a boolean variable, %s is not '
                             'boolean.') % var.name)
        self.conditional_write = var

    def get_value(self):
        return self.device.get_value(self)

    def set_value(self, value):
        self.device.fill_with_array(self, value)

    def get_len(self):
        return self.size

    def get_addressable_value(self, name, group):
        return VariableView(name=name, variable=self, group=group, unit=None)

    def get_addressable_value_with_unit(self, name, group):
        return VariableView(name=name, variable=self, group=group,
                            unit=self.unit)


class DynamicArrayVariable(ArrayVariable):
    '''
    An object providing information about a model variable stored in a dynamic
    array (used in `Synapses`). Most of the time `Variables.add_dynamic_array`
    should be used instead of instantiating this class directly.

    Parameters
    ----------
    name : 'str'
        The name of the variable. Note that this refers to the *original*
        name in the owning group. The same variable may be known under other
        names in other groups (e.g. the variable ``v`` of a `NeuronGroup` is
        known as ``v_post`` in a `Synapse` connecting to the group).
    unit : `Unit`
        The unit of the variable
    owner : `Nameable`
        The object that "owns" this variable, e.g. the `NeuronGroup` or
        `Synapses` object that declares the variable in its model equations.
    size : int or tuple of int
        The (initial) size of the variable.
    device : `Device`
        The device responsible for the memory access.
    dtype : `dtype`, optional
        The dtype used for storing the variable. If none is given, defaults
        to `core.default_float_dtype`.
    constant : bool, optional
        Whether the variable's value is constant during a run.
        Defaults to ``False``.
    needs_reference_update : bool, optional
        Whether the code objects need a new reference to the underlying data at
        every time step. This should be set if the size of the array can be
        changed by other code objects. Defaults to ``False``.
    scalar : bool, optional
        Whether this array is a 1-element array that should be treated like a
        scalar (e.g. for a single delay value across synapses). Defaults to
        ``False``.
    read_only : bool, optional
        Whether this is a read-only variable, i.e. a variable that is set
        internally and cannot be changed by the user. Defaults
        to ``False``.
    unique : bool, optional
        Whether the values in this array are all unique. This information is
        only important for variables used as indices and does not have to
        reflect the actual contents of the array but only the possibility of
        non-uniqueness (e.g. synaptic indices are always unique but the
        corresponding pre- and post-synaptic indices are not). Defaults to
        ``False``.
    '''

    def __init__(self, name, unit, owner, size, device, dtype=None,
                 constant=False, needs_reference_update=False,
                 resize_along_first=False, scalar=False, read_only=False,
                 unique=False):

        if isinstance(size, int):
            dimensions = 1
        else:
            dimensions = len(size)

        #: The number of dimensions
        self.dimensions = dimensions

        if constant and needs_reference_update:
            raise ValueError('A variable cannot be constant and '
                             'need reference updates')
        #: Whether this variable needs an update of the reference to the
        #: underlying data whenever it is passed to a code object
        self.needs_reference_update = needs_reference_update

        #: Whether this array will be only resized along the first dimension
        self.resize_along_first = resize_along_first

        super(DynamicArrayVariable, self).__init__(unit=unit,
                                                   owner=owner,
                                                   name=name,
                                                   size=size,
                                                   device=device,
                                                   constant=constant,
                                                   dtype=dtype,
                                                   scalar=scalar,
                                                   dynamic=True,
                                                   read_only=read_only,
                                                   unique=unique)

    def resize(self, new_size):
        '''
        Resize the dynamic array. Calls `self.device.resize` to do the
        actual resizing.

        Parameters
        ----------
        new_size : int or tuple of int
            The new size.
        '''
        if self.resize_along_first:
            self.device.resize_along_first(self, new_size)
        else:
            self.device.resize(self, new_size)

        self.size = new_size



class Subexpression(Variable):
    '''
    An object providing information about a named subexpression in a model.
    Most of the time `Variables.add_subexpression` should be used instead of
    instantiating this class directly.

    Parameters
    ----------
    name : str
        The name of the subexpression.
    unit : `Unit`
        The unit of the subexpression.
    owner : `Group`
        The group to which the expression refers.
    expr : str
        The subexpression itself.
    device : `Device`
        The device responsible for the memory access.
    dtype : `dtype`, optional
        The dtype used for the expression. Defaults to
        `core.default_float_dtype`.
    scalar: bool, optional
        Whether this is an expression only referring to scalar variables.
        Defaults to ``False``
    '''
    def __init__(self, name, unit, owner, expr, device, dtype=None,
                 scalar=False):
        super(Subexpression, self).__init__(unit=unit, owner=owner,
                                            name=name, dtype=dtype,
                                            scalar=scalar,
                                            constant=False, read_only=True)

        #: The `Device` responsible for memory access
        self.device = device

        #: The expression defining the subexpression
        self.expr = expr.strip()

        if scalar:
            from brian2.parsing.sympytools import str_to_sympy
            # We check here if the corresponding sympy expression contains a
            # reference to _vectorisation_idx which indicates that an implicitly
            # vectorized function (e.g. rand() ) has been used. We do not allow
            # this since it would lead to incorrect results when substituted into
            # vector equations
            sympy_expr = str_to_sympy(self.expr)
            if sympy.Symbol('_vectorisation_idx') in sympy_expr.atoms():
                raise SyntaxError(('The scalar subexpression %s refers to an '
                                   'implicitly vectorized function -- this is '
                                   'not allowed since it leads to different '
                                   'interpretations of this subexpression '
                                   'depending on whether it is used in a '
                                   'scalar or vector context.') % name)

        #: The identifiers used in the expression
        self.identifiers = get_identifiers(expr)

    def get_addressable_value(self, name, group):
        return VariableView(name=name, variable=self, group=group, unit=None)

    def get_addressable_value_with_unit(self, name, group):
        return VariableView(name=name, variable=self, group=group,
                            unit=self.unit)

    def __contains__(self, var):
        return var in self.identifiers

    def __repr__(self):
        description = ('<{classname}(name={name}, unit={unit}, dtype={dtype}, '
                       'expr={expr}, owner=<{owner}>)>')
        return description.format(classname=self.__class__.__name__,
                                  name=repr(self.name),
                                  unit=repr(self.unit),
                                  dtype=repr(self.dtype),
                                  expr=repr(self.expr),
                                  owner=self.owner.name)


# ------------------------------------------------------------------------------
# Classes providing views on variables and storing variables information
# ------------------------------------------------------------------------------
class LinkedVariable(object):
    '''
    A simple helper class to make linking variables explicit. Users should use
    `linked_var` instead.

    Parameters
    ----------
    group : `Group`
        The group through which the `variable` is accessed (not necessarily the
        same as ``variable.owner``.
    name : str
        The name of `variable` in `group` (not necessarily the same as
         ``variable.name``).
    variable : `Variable`
        The variable that should be linked.
    index : str or `ndarray`, optional
        An indexing array (or the name of a state variable), providing a mapping
        from the entries in the link source to the link target.
    '''
    def __init__(self, group, name, variable, index=None):
        self.group = group
        self.name = name
        self.variable = variable
        self.index = index


def linked_var(group_or_variable, name=None, index=None):
    '''
    Represents a link target for setting a linked variable.

    Parameters
    ----------
    group_or_variable : `NeuronGroup` or `VariableView`
        Either a reference to the target `NeuronGroup` (e.g. ``G``) or a direct
        reference to a `VariableView` object (e.g. ``G.v``). In case only the
        group is specified, `name` has to be specified as well.
    name : str, optional
        The name of the target variable, necessary if `group_or_variable` is a
        `NeuronGroup`.
    index : str or `ndarray`, optional
        An indexing array (or the name of a state variable), providing a mapping
        from the entries in the link source to the link target.

    Examples
    --------
    >>> from brian2 import *
    >>> G1 = NeuronGroup(10, 'dv/dt = -v / (10*ms) : volt')
    >>> G2 = NeuronGroup(10, 'v : volt (linked)')
    >>> G2.v = linked_var(G1, 'v')
    >>> G2.v = linked_var(G1.v)  # equivalent
    '''
    if isinstance(group_or_variable, VariableView):
        if name is not None:
            raise ValueError(('Cannot give a variable and a variable name at '
                              'the same time.'))
        return LinkedVariable(group_or_variable.group,
                              group_or_variable.name,
                              group_or_variable.variable, index=index)
    elif name is None:
        raise ValueError('Need to provide a variable name')
    else:
        return LinkedVariable(group_or_variable,
                              name,
                              group_or_variable.variables[name], index=index)


class VariableView(object):
    '''
    A view on a variable that allows to treat it as an numpy array while
    allowing special indexing (e.g. with strings) in the context of a `Group`.

    Parameters
    ----------
    name : str
        The name of the variable (not necessarily the same as ``variable.name``).
    variable : `Variable`
        The variable description.
    group : `Group`
        The group through which the variable is accessed (not necessarily the
        same as `variable.owner`).
    unit : `Unit`, optional
        The unit to be used for the variable, should be `None` when a variable
         is accessed without units (e.g. when accessing ``G.var_``).
    '''
    def __init__(self, name, variable, group, unit=None):
        self.name = name
        self.variable = variable
        self.index_var_name = group.variables.indices[name]
        if self.index_var_name in ('_idx', '0'):
            self.index_var = self.index_var_name
        else:
            self.index_var = group.variables[self.index_var_name]

        if isinstance(variable, Subexpression):
            # For subexpressions, we *always* have to go via codegen to get
            # their value -- since we cannot do this without the group, we
            # hold a strong reference
            self.group = group
        else:
            # For state variable arrays, we can do most access without the full
            # group, using the indexing reference below. We therefore only keep
            # a weak reference to the group.
            self.group = weakproxy_with_fallback(group)
        self.group_name = group.name
        # We keep a strong reference to the `Indexing` object so that basic
        # indexing is still possible, even if the group no longer exists
        self.indexing = self.group._indices
        self.unit = unit

    @property
    def dim(self):
        '''
        The dimensions of this variable.
        '''
        return self.unit.dim

    def get_item(self, item, level=0, namespace=None):
        '''
        Get the value of this variable. Called by `__getitem__`.

        Parameters
        ----------
        item : slice, `ndarray` or string
            The index for the setting operation
        level : int, optional
            How much farther to go up in the stack to find the implicit
            namespace (if used, see `run_namespace`).
        namespace : dict-like, optional
            An additional namespace that is used for variable lookup (if not
            defined, the implicit namespace of local variables is used).
        '''
        if isinstance(item, basestring):
            # Check whether the group still exists to give a more meaningful
            # error message if it does not
            try:
                self.group.name
            except ReferenceError:
                raise ReferenceError(('Cannot use string expressions, the '
                                      'group "%s", providing the context for '
                                      'the expression, no longer exists. '
                                      'Consider holding an explicit reference '
                                      'to it to keep it '
                                      'alive.') % self.group_name)
            values = self.get_with_expression(item,
                                              level=level+1,
                                              run_namespace=namespace)
        else:
            if isinstance(self.variable, Subexpression):
                values = self.get_subexpression_with_index_array(item,
                                                                 level=level+1,
                                                                 run_namespace=namespace)
            else:
                values = self.get_with_index_array(item)

        if self.unit is None:
            return values
        else:
            return Quantity(values, self.unit.dimensions)

    def __getitem__(self, item):
        return self.get_item(item, level=1)

    def set_item(self, item, value, level=0, namespace=None):
        '''
        Set this variable. This function is called by `__setitem__` but there
        is also a situation where it should be called directly: if the context
        for string-based expressions is higher up in the stack, this function
        allows to set the `level` argument accordingly.

        Parameters
        ----------
        item : slice, `ndarray` or string
            The index for the setting operation
        value : `Quantity`, `ndarray` or number
            The value for the setting operation
        level : int, optional
            How much farther to go up in the stack to find the implicit
            namespace (if used, see `run_namespace`).
        namespace : dict-like, optional
            An additional namespace that is used for variable lookup (if not
            defined, the implicit namespace of local variables is used).
        '''
        variable = self.variable
        if variable.read_only:
            raise TypeError('Variable %s is read-only.' % self.name)

        # The second part is equivalent to item == slice(None) but formulating
        # it this way prevents a FutureWarning if one of the elements is a
        # numpy array
        if isinstance(item, slice) and (item.start is None and
                                        item.stop is None and
                                        item.step is None):
            item = 'True'

        check_units = self.unit is not None

        # Both index and values are strings, use a single code object do deal
        # with this situation
        if isinstance(value, basestring) and isinstance(item, basestring):
            self.set_with_expression_conditional(item, value,
                                                 check_units=check_units,
                                                 level=level+1,
                                                 run_namespace=namespace)
        elif isinstance(item, basestring):
            try:
                float(value)  # only checks for the exception
                try:
                    # length-1 arrays are also convertible to float, but we
                    # don't want the repr used later to be something like
                    # array([...]).
                    value = value[0]
                except (IndexError, TypeError):
                    # was scalar already apparently
                    pass
            except (TypeError, ValueError):
                if item != 'True':
                    raise TypeError('When setting a variable based on a string '
                                    'index, the value has to be a string or a '
                                    'scalar.')

            if item == 'True':
                # We do not want to go through code generation for runtime
                    self.set_with_index_array(slice(None), value,
                                              check_units=check_units)
            else:
                self.set_with_expression_conditional(item,
                                                     repr(value),
                                                     check_units=check_units,
                                                     level=level+1,
                                                     run_namespace=namespace)
        elif isinstance(value, basestring):
            self.set_with_expression(item, value,
                                     check_units=check_units,
                                     level=level+1,
                                     run_namespace=namespace)
        else:  # No string expressions involved
            self.set_with_index_array(item, value,
                                      check_units=check_units)

    def __setitem__(self, item, value):
        self.set_item(item, value, level=1)

    @device_override('variableview_set_with_expression')
    def set_with_expression(self, item, code, check_units=True, level=0,
                            run_namespace=None):
        '''
        Sets a variable using a string expression. Is called by
        `VariableView.set_item` for statements such as
        ``S.var[:, :] = 'exp(-abs(i-j)/space_constant)*nS'``

        Parameters
        ----------
        item : `ndarray`
            The indices for the variable (in the context of this `group`).
        code : str
            The code that should be executed to set the variable values.
            Can contain references to indices, such as `i` or `j`
        check_units : bool, optional
            Whether to check the units of the expression.
        level : int, optional
            How much farther to go up in the stack to find the implicit
            namespace (if used, see `run_namespace`).
        run_namespace : dict-like, optional
            An additional namespace that is used for variable lookup (if not
            defined, the implicit namespace of local variables is used).
        '''
        # Some fairly complicated code to raise a warning in ambiguous
        # situations, when indexing with a group. For example, in:
        #   group.v[subgroup] =  'i'
        # the index 'i' is the index of 'group' ("absolute index") and not of
        # subgroup ("relative index")
        if hasattr(item, 'variables') or (isinstance(item, tuple)
                                          and any(hasattr(one_item, 'variables')
                                                  for one_item in item)):
            # Determine the variables that are used in the expression
            from brian2.codegen.translation import get_identifiers_recursively
            identifiers = get_identifiers_recursively([code],
                                                      self.group.variables)
            variables = self.group.resolve_all(identifiers, [],
                                               run_namespace=run_namespace,
                                               level=level+2)
            if not isinstance(item, tuple):
                index_groups = [item]
            else:
                index_groups = item

            for varname, var in variables.iteritems():
                for index_group in index_groups:
                    if not hasattr(index_group, 'variables'):
                        continue
                    if varname in index_group.variables or var.name in index_group.variables:
                        indexed_var = index_group.variables.get(varname,
                                                                index_group.variables.get(var.name))
                        if not indexed_var is var:
                            logger.warn(('The string expression used for setting '
                                         '{varname} refers to {referred_var} which '
                                         'might be ambiguous. It will be '
                                         'interpreted as referring to '
                                         '{referred_var} in {group}, not as '
                                         'a variable of a group used for '
                                         'indexing.').format(varname=self.name,
                                                             referred_var=varname,
                                                             group=self.group.name,
                                                             index_group=index_group.name),
                                        'ambiguous_string_expression')
                            break  # no need to warn more than once for a variable

        indices = self.indexing(item)
        abstract_code = self.name + ' = ' + code
        variables = Variables(None)
        variables.add_array('_group_idx', unit=Unit(1),
                            size=len(indices), dtype=np.int32)
        variables['_group_idx'].set_value(indices)

        # TODO: Have an additional argument to avoid going through the index
        # array for situations where iterate_all could be used
        from brian2.codegen.codeobject import create_runner_codeobj
        from brian2.devices.device import get_default_codeobject_class
        codeobj = create_runner_codeobj(self.group,
                                        abstract_code,
                                        'group_variable_set',
                                        additional_variables=variables,
                                        check_units=check_units,
                                        level=level+2,
                                        run_namespace=run_namespace,
                                        codeobj_class=get_default_codeobject_class('codegen.string_expression_target'))
        codeobj()

    @device_override('variableview_set_with_expression_conditional')
    def set_with_expression_conditional(self, cond,
                                        code, check_units=True, level=0,
                                        run_namespace=None):
        '''
        Sets a variable using a string expression and string condition. Is
        called by `VariableView.set_item` for statements such as
        ``S.var['i!=j'] = 'exp(-abs(i-j)/space_constant)*nS'``

        Parameters
        ----------
        cond : str
            The string condition for which the variables should be set.
        code : str
            The code that should be executed to set the variable values.
        check_units : bool, optional
            Whether to check the units of the expression.
        level : int, optional
            How much farther to go up in the stack to find the implicit
            namespace (if used, see `run_namespace`).
        run_namespace : dict-like, optional
            An additional namespace that is used for variable lookup (if not
            defined, the implicit namespace of local variables is used).
        '''
        variable = self.variable
        if variable.scalar and cond != 'True':
            raise IndexError(('Cannot conditionally set the scalar variable '
                              '%s.') % self.name)
        abstract_code_cond = '_cond = '+cond
        abstract_code = self.name + ' = ' + code
        variables = Variables(None)
        variables.add_auxiliary_variable('_cond', unit=Unit(1), dtype=np.bool)
        from brian2.codegen.codeobject import create_runner_codeobj
        # TODO: Have an additional argument to avoid going through the index
        # array for situations where iterate_all could be used
        from brian2.devices.device import get_default_codeobject_class
        codeobj = create_runner_codeobj(self.group,
                                        {'condition': abstract_code_cond,
                                         'statement': abstract_code},
                                        'group_variable_set_conditional',
                                        additional_variables=variables,
                                        check_units=check_units,
                                        level=level+2,
                                        run_namespace=run_namespace,
                                        codeobj_class=get_default_codeobject_class('codegen.string_expression_target'))
        codeobj()

    @device_override('variableview_get_with_expression')
    def get_with_expression(self, code, level=0, run_namespace=None):
        '''
        Gets a variable using a string expression. Is called by
        `VariableView.get_item` for statements such as
        ``print G.v['g_syn > 0']``.

        Parameters
        ----------
        code : str
            An expression that states a condition for elements that should be
            selected. Can contain references to indices, such as ``i`` or ``j``
            and to state variables. For example: ``'i>3 and v>0*mV'``.
        level : int, optional
            How much farther to go up in the stack to find the implicit
            namespace (if used, see `run_namespace`).
        run_namespace : dict-like, optional
            An additional namespace that is used for variable lookup (if not
            defined, the implicit namespace of local variables is used).
        '''
        variable = self.variable
        if variable.scalar:
            raise IndexError(('Cannot access the variable %s with a '
                              'string expression, it is a scalar '
                              'variable.') % self.name)
        # Add the recorded variable under a known name to the variables
        # dictionary. Important to deal correctly with
        # the type of the variable in C++
        variables = Variables(None)
        variables.add_auxiliary_variable('_variable', unit=variable.unit,
                                         dtype=variable.dtype,
                                         scalar=variable.scalar)
        variables.add_auxiliary_variable('_cond', unit=Unit(1), dtype=np.bool)

        abstract_code = '_variable = ' + self.name + '\n'
        abstract_code += '_cond = ' + code
        from brian2.codegen.codeobject import create_runner_codeobj
        from brian2.devices.device import get_default_codeobject_class
        codeobj = create_runner_codeobj(self.group,
                                        abstract_code,
                                        'group_variable_get_conditional',
                                        additional_variables=variables,
                                        level=level+2,
                                        run_namespace=run_namespace,
                                        codeobj_class=get_default_codeobject_class('codegen.string_expression_target')
                                        )
        return codeobj()

    @device_override('variableview_get_with_index_array')
    def get_with_index_array(self, item):
        variable = self.variable
        if variable.scalar:
            if not (isinstance(item, slice) and item == slice(None)):
                raise IndexError(('Illegal index for variable %s, it is a '
                                  'scalar variable.') % self.name)
            indices = 0
        elif (isinstance(item, slice) and item == slice(None)
              and self.index_var == '_idx'):
            indices = slice(None)
        else:
            indices = self.indexing(item, self.index_var)

        return variable.get_value()[indices]

    @device_override('variableview_get_subexpression_with_index_array')
    def get_subexpression_with_index_array(self, item, level=0, run_namespace=None):
        variable = self.variable
        if variable.scalar:
            if not (isinstance(item, slice) and item == slice(None)):
                raise IndexError(('Illegal index for variable %s, it is a '
                                  'scalar variable.') % self.name)
            indices = np.array(0)
        else:
            indices = self.indexing(item, self.index_var)

        # For "normal" variables, we can directly access the underlying data
        # and use the usual slicing syntax. For subexpressions, however, we
        # have to evaluate code for the given indices
        variables = Variables(None, default_index='_group_index')
        variables.add_auxiliary_variable('_variable',
                                         unit=variable.unit,
                                         dtype=variable.dtype,
                                         scalar=variable.scalar)
        if indices.shape == ():
            single_index = True
            indices = np.array([indices])
        else:
            single_index = False
        variables.add_array('_group_idx', unit=Unit(1),
                            size=len(indices), dtype=np.int32)
        variables['_group_idx'].set_value(indices)
        # Force the use of this variable as a replacement for the original
        # index variable
        using_orig_index = [varname for varname, index in self.group.variables.indices.iteritems()
                            if index == self.index_var_name and index != '0']
        for varname in using_orig_index:
            variables.indices[varname] = '_idx'

        abstract_code = '_variable = ' + self.name + '\n'
        from brian2.codegen.codeobject import create_runner_codeobj
        from brian2.devices.device import get_default_codeobject_class
        codeobj = create_runner_codeobj(self.group,
                                        abstract_code,
                                        'group_variable_get',
                                        # Setting the user code to an empty
                                        # string suppresses warnings if the
                                        # subexpression refers to variable
                                        # names that are also present in the
                                        # local namespace
                                        user_code='',
                                        needed_variables=['_group_idx'],
                                        additional_variables=variables,
                                        level=level+2,
                                        run_namespace=run_namespace,
                                        codeobj_class=get_default_codeobject_class('codegen.string_expression_target')
        )
        result = codeobj()
        if single_index and not variable.scalar:
            return result[0]
        else:
            return result

    @device_override('variableview_set_with_index_array')
    def set_with_index_array(self, item, value, check_units):
        variable = self.variable
        if check_units:
            fail_for_dimension_mismatch(variable.unit, value,
                                        'Incorrect unit for setting variable %s' % self.name)
        if variable.scalar:
            if not (isinstance(item, slice) and item == slice(None)):
                raise IndexError(('Illegal index for variable %s, it is a '
                                  'scalar variable.') % self.name)
            indices = 0
        elif (isinstance(item, slice) and item == slice(None)
              and self.index_var == '_idx'):
            indices = slice(None)
        else:
            indices = self.indexing(item, self.index_var)

            q = Quantity(value, copy=False)
            if len(q.shape):
                if not len(q.shape) == 1 or len(q) != 1 and len(q) != len(indices):
                    raise ValueError(('Provided values do not match the size '
                                      'of the indices, '
                                      '%d != %d.') % (len(q),
                                                      len(indices)))
        variable.get_value()[indices] = value

    # Allow some basic calculations directly on the ArrayView object
    def __array__(self, dtype=None):
        try:
            # This will fail for subexpressions that refer to external
            # parameters
            value = self[:]
        except ValueError:
            raise ValueError(('Cannot get the values for variable {var}. If it '
                              'is a subexpression referring to external '
                              'variables, use "group.{var}[:]" instead of '
                              '"group.{var}"'.format(var=self.variable.name)))
        return np.asanyarray(self[:], dtype=dtype)

    def __array_prepare__(self, array, context=None):
        if self.unit is None:
            return array
        else:
            this = self[:]
            if isinstance(this, Quantity):
                return Quantity.__array_prepare__(this, array,
                                                  context=context)
            else:
                return array

    def __array_wrap__(self, out_arr, context=None):
        if self.unit is None:
            return out_arr
        else:
            this = self[:]
            if isinstance(this, Quantity):
                return Quantity.__array_wrap__(self[:], out_arr,
                                               context=context)
            else:
                return out_arr

    def __len__(self):
        return len(self.get_item(slice(None), level=1))

    def __neg__(self):
        return -self.get_item(slice(None), level=1)

    def __pos__(self):
        return self.get_item(slice(None), level=1)

    def __add__(self, other):
        return self.get_item(slice(None), level=1) + np.asanyarray(other)

    def __radd__(self, other):
        return np.asanyarray(other) + self.get_item(slice(None), level=1)

    def __sub__(self, other):
        return self.get_item(slice(None), level=1) - np.asanyarray(other)

    def __rsub__(self, other):
        return np.asanyarray(other) - self.get_item(slice(None), level=1)

    def __mul__(self, other):
        return self.get_item(slice(None), level=1) * np.asanyarray(other)

    def __rmul__(self, other):
        return np.asanyarray(other) * self.get_item(slice(None), level=1)

    def __div__(self, other):
        return self.get_item(slice(None), level=1) / np.asanyarray(other)

    def __truediv__(self, other):
        return self.get_item(slice(None), level=1) / np.asanyarray(other)

    def __floordiv__(self, other):
        return self.get_item(slice(None), level=1) // np.asanyarray(other)

    def __rdiv__(self, other):
        return np.asanyarray(other) / self.get_item(slice(None), level=1)

    def __rtruediv__(self, other):
        return np.asanyarray(other) / self.get_item(slice(None), level=1)

    def __rfloordiv__(self, other):
        return np.asanyarray(other) // self.get_item(slice(None), level=1)

    def __iadd__(self, other):
        if isinstance(other, basestring):
            raise TypeError(('In-place modification with strings not '
                             'supported. Use group.var = "var + expression" '
                             'instead of group.var += "expression".'))
        elif isinstance(self.variable, Subexpression):
            raise TypeError('Cannot assign to a subexpression.')
        else:
            rhs = self[:] + np.asanyarray(other)
        self[:] = rhs
        return self

    def __isub__(self, other):
        if isinstance(other, basestring):
            raise TypeError(('In-place modification with strings not '
                             'supported. Use group.var = "var - expression" '
                             'instead of group.var -= "expression".'))
        elif isinstance(self.variable, Subexpression):
            raise TypeError('Cannot assign to a subexpression.')
        else:
            rhs = self[:] - np.asanyarray(other)
        self[:] = rhs
        return self

    def __imul__(self, other):
        if isinstance(other, basestring):
            raise TypeError(('In-place modification with strings not '
                             'supported. Use group.var = "var * expression" '
                             'instead of group.var *= "expression".'))
        elif isinstance(self.variable, Subexpression):
            raise TypeError('Cannot assign to a subexpression.')
        else:
            rhs = self[:] * np.asanyarray(other)
        self[:] = rhs
        return self

    def __idiv__(self, other):
        if isinstance(other, basestring):
            raise TypeError(('In-place modification with strings not '
                             'supported. Use group.var = "var / expression" '
                             'instead of group.var /= "expression".'))
        elif isinstance(self.variable, Subexpression):
            raise TypeError('Cannot assign to a subexpression.')
        else:
            rhs = self[:] / np.asanyarray(other)
        self[:] = rhs
        return self

    # Also allow logical comparisons

    def __eq__(self, other):
        return self.get_item(slice(None), level=1) == np.asanyarray(other)

    def __ne__(self, other):
        return self.get_item(slice(None), level=1) != np.asanyarray(other)

    def __lt__(self, other):
        return self.get_item(slice(None), level=1) < np.asanyarray(other)

    def __le__(self, other):
        return self.get_item(slice(None), level=1) <= np.asanyarray(other)

    def __gt__(self, other):
        return self.get_item(slice(None), level=1) > np.asanyarray(other)

    def __ge__(self, other):
        return self.get_item(slice(None), level=1) >= np.asanyarray(other)

    def __repr__(self):
        varname = self.name
        if self.unit is None:
            varname += '_'

        if self.variable.scalar:
            dim = self.unit.dim if self.unit is not None else DIMENSIONLESS
            values = repr(Quantity(self.variable.get_value().item(),
                                   dim=dim))
        else:
            try:
                # This will fail for subexpressions that refer to external
                # parameters
                values = repr(self[:])
            except KeyError:
                values = ('[Subexpression refers to external parameters. Use '
                          '"group.{var}[:]"]').format(var=self.variable.name)

        return '<%s.%s: %s>' % (self.group_name, varname,
                                 values)

    # Get access to some basic properties of the underlying array
    @property
    def shape(self):
        return self.get_item(slice(None), level=1).shape

    @property
    def dtype(self):
        return self.get_item(slice(None), level=1).dtype



class Variables(collections.Mapping):
    '''
    A container class for storing `Variable` objects. Instances of this class
    are used as the `Group.variables` attribute and can be accessed as
    (read-only) dictionaries.

    Parameters
    ----------
    owner : `Nameable`
        The object (typically a `Group`) "owning" the variables.
    default_index : str, optional
        The index to use for the variables (only relevant for `ArrayVariable`
        and `DynamicArrayVariable`). Defaults to ``'_idx'``.
    '''

    def __init__(self, owner, default_index='_idx'):
        #: A reference to the `Group` owning these variables
        self.owner = weakproxy_with_fallback(owner)
        # The index that is used for arrays if no index is given explicitly
        self.default_index = default_index

        # We do the import here to avoid a circular dependency.
        from brian2.devices.device import get_device
        self.device = get_device()

        self._variables = {}
        #: A dictionary given the index name for every array name
        self.indices = collections.defaultdict(functools.partial(str, default_index))
        # Note that by using functools.partial (instead of e.g. a lambda
        # function) above, this object remains pickable.

    def __getitem__(self, item):
        return self._variables[item]

    def __len__(self):
        return len(self._variables)

    def __iter__(self):
        return iter(self._variables)

    def _add_variable(self, name, var, index=None):
        if name in self._variables:
            raise KeyError(('The name "%s" is already present in the variables'
                            ' dictionary.') % name)
        #TODO: do some check for the name, part of it has to be device-specific
        self._variables[name] = var

        if isinstance(var, ArrayVariable):
            # Tell the device to actually create the array (or note it down for
            # later code generation in standalone).
            self.device.add_array(var)

        if getattr(var, 'scalar', False):
            if index not in (None, '0'):
                raise ValueError('Cannot set an index for a scalar variable')
            self.indices[name] = '0'

        if index is not None:
            self.indices[name] = index

    def add_array(self, name, unit, size, values=None, dtype=None,
                  constant=False, read_only=False, scalar=False, unique=False,
                  index=None):
        '''
        Add an array (initialized with zeros).

        Parameters
        ----------
        name : str
            The name of the variable.
        unit : `Unit`
            The unit of the variable
        size : int
            The size of the array.
        values : `ndarray`, optional
            The values to initalize the array with. If not specified, the array
            is initialized to zero.
        dtype : `dtype`, optional
            The dtype used for storing the variable. If none is given, defaults
            to `core.default_float_dtype`.
        constant : bool, optional
            Whether the variable's value is constant during a run.
            Defaults to ``False``.
        scalar : bool, optional
            Whether this is a scalar variable. Defaults to ``False``, if set to
            ``True``, also implies that `size` equals 1.
        read_only : bool, optional
            Whether this is a read-only variable, i.e. a variable that is set
            internally and cannot be changed by the user. Defaults
            to ``False``.
        index : str, optional
            The index to use for this variable. Defaults to
            `Variables.default_index`.
        unique : bool, optional
            See `ArrayVariable`. Defaults to ``False``.
        '''
        if np.asanyarray(size).shape == ():
            # We want a basic Python type for the size instead of something
            # like numpy.int64
            size = int(size)
        var = ArrayVariable(name=name, unit=unit, owner=self.owner,
                            device=self.device, size=size,
                            dtype=dtype,
                            constant=constant,
                            scalar=scalar,
                            read_only=read_only,
                            unique=unique)
        self._add_variable(name, var, index)
        # This could be avoided, but we currently need it so that standalone
        # allocates the memory
        self.device.init_with_zeros(var, dtype)
        if values is not None:
            if scalar:
                if np.asanyarray(values).shape != ():
                    raise ValueError('Need a scalar value.')
                self.device.fill_with_array(var, values)
            else:
                if len(values) != size:
                    raise ValueError(('Size of the provided values does not match '
                                      'size: %d != %d') % (len(values), size))
                self.device.fill_with_array(var, values)

    def add_arrays(self, names, unit, size, values=None, dtype=None,
                  constant=False, read_only=False, scalar=False, unique=False,
                  index=None):
        '''
        Adds several arrays (initialized with zeros) with the same attributes
        (size, units, etc.).

        Parameters
        ----------
        names : list of str
            The names of the variable.
        unit : `Unit`
            The unit of the variables
        size : int
            The sizes of the arrays.
        dtype : `dtype`, optional
            The dtype used for storing the variables. If none is given, defaults
            to `core.default_float_dtype`.
        constant : bool, optional
            Whether the variables' values are constant during a run.
            Defaults to ``False``.
        scalar : bool, optional
            Whether these are scalar variables. Defaults to ``False``, if set to
            ``True``, also implies that `size` equals 1.
        read_only : bool, optional
            Whether these are read-only variables, i.e. variables that are set
            internally and cannot be changed by the user. Defaults
            to ``False``.
        index : str, optional
            The index to use for these variables. Defaults to
            `Variables.default_index`.
        unique : bool, optional
            See `ArrayVariable`. Defaults to ``False``.
        '''
        for name in names:
            self.add_array(name, unit=unit, size=size, dtype=dtype,
                           constant=constant, read_only=read_only,
                           scalar=scalar, unique=unique, index=index)

    def add_dynamic_array(self, name, unit, size, values=None, dtype=None,
                          constant=False, needs_reference_update=False,
                          resize_along_first=False, read_only=False,
                          unique=False, scalar=False, index=None):
        '''
        Add a dynamic array.

        Parameters
        ----------
        name : str
            The name of the variable.
        unit : `Unit`
            The unit of the variable
        size : int or tuple of int
            The (initital) size of the array.
        values : `ndarray`, optional
            The values to initalize the array with. If not specified, the array
            is initialized to zero.
        dtype : `dtype`, optional
            The dtype used for storing the variable. If none is given, defaults
            to `core.default_float_dtype`.
        constant : bool, optional
            Whether the variable's value is constant during a run.
            Defaults to ``False``.
        needs_reference_update : bool, optional
            Whether the code objects need a new reference to the underlying data at
            every time step. This should be set if the size of the array can be
            changed by other code objects. Defaults to ``False``.
        scalar : bool, optional
            Whether this is a scalar variable. Defaults to ``False``, if set to
            ``True``, also implies that `size` equals 1.
        read_only : bool, optional
            Whether this is a read-only variable, i.e. a variable that is set
            internally and cannot be changed by the user. Defaults
            to ``False``.
        index : str, optional
            The index to use for this variable. Defaults to
            `Variables.default_index`.
        unique : bool, optional
            See `DynamicArrayVariable`. Defaults to ``False``.
        '''
        var = DynamicArrayVariable(name=name, unit=unit, owner=self.owner,
                                   device=self.device,
                                   size=size, dtype=dtype,
                                   constant=constant,
                                   needs_reference_update=needs_reference_update,
                                   resize_along_first=resize_along_first,
                                   scalar=scalar,
                                   read_only=read_only, unique=unique)
        self._add_variable(name, var, index)
        if np.prod(size) > 0:
            self.device.resize(var, size)
        if values is None and np.prod(size) > 0:
            self.device.init_with_zeros(var, dtype)
        elif values is not None:
            if len(values) != size:
                raise ValueError(('Size of the provided values does not match '
                                  'size: %d != %d') % (len(values), size))
            if np.prod(size) > 0:
                self.device.fill_with_array(var, values)

    def add_arange(self, name, size, start=0, dtype=np.int32, constant=True,
                   read_only=True, unique=True, index=None):
        '''
        Add an array, initialized with a range of integers.

        Parameters
        ----------
        name : str
            The name of the variable.
        size : int
            The size of the array.
        start : int
            The start value of the range.
        dtype : `dtype`, optional
            The dtype used for storing the variable. If none is given, defaults
            to `np.int32`.
        constant : bool, optional
            Whether the variable's value is constant during a run.
            Defaults to ``True``.
        read_only : bool, optional
            Whether this is a read-only variable, i.e. a variable that is set
            internally and cannot be changed by the user. Defaults
            to ``True``.
        index : str, optional
            The index to use for this variable. Defaults to
            `Variables.default_index`.
        unique : bool, optional
            See `ArrayVariable`. Defaults to ``True`` here.
        '''
        self.add_array(name=name, unit=Unit(1), size=size, dtype=dtype,
                       constant=constant, read_only=read_only, unique=unique,
                       index=index)
        self.device.init_with_arange(self._variables[name], start, dtype=dtype)


    def add_constant(self, name, unit, value):
        '''
        Add a scalar constant (e.g. the number of neurons `N`).

        Parameters
        ----------
        name : str
            The name of the variable
        unit : `Unit`
            The unit of the variable. Note that the variable itself (as referenced
            by value) should never have units attached.
        value: reference to the variable value
            The value of the constant.
        '''
        var = Constant(name=name, unit=unit, owner=self.owner, value=value)
        self._add_variable(name, var)

    def add_subexpression(self, name, unit, expr, dtype=None, scalar=False,
                          index=None):
        '''
        Add a named subexpression.

        Parameters
        ----------
        name : str
            The name of the subexpression.
        unit : `Unit`
            The unit of the subexpression.
        expr : str
            The subexpression itself.
        dtype : `dtype`, optional
            The dtype used for the expression. Defaults to
            `core.default_float_dtype`.
        scalar : bool, optional
            Whether this is an expression only referring to scalar variables.
            Defaults to ``False``
        index : str, optional
            The index to use for this variable. Defaults to
            `Variables.default_index`.
        '''
        var = Subexpression(name=name, unit=unit, expr=expr, owner=self.owner,
                            dtype=dtype, device=self.device, scalar=scalar)
        self._add_variable(name, var, index=index)

    def add_auxiliary_variable(self, name, unit, dtype=None, scalar=False):
        '''
        Add an auxiliary variable (most likely one that is added automatically
        to abstract code, e.g. ``_cond`` for a threshold condition),
        specifying its type and unit for code generation.

        Parameters
        ----------
        name : str
            The name of the variable
        unit : `Unit`
            The unit of the variable.
        dtype : `dtype`, optional
            The dtype used for storing the variable. If none is given, defaults
            to `core.default_float_dtype`.
        scalar : bool, optional
            Whether the variable is a scalar value (``True``) or vector-valued,
            e.g. defined for every neuron (``False``). Defaults to ``False``.
        '''
        var = AuxiliaryVariable(name=name, unit=unit, dtype=dtype,
                                scalar=scalar)
        self._add_variable(name, var)


    def add_referred_subexpression(self, name, group, subexpr, index):
        identifiers = subexpr.identifiers
        substitutions = {}
        for identifier in identifiers:
            if not identifier in subexpr.owner.variables:
                # external variable --> nothing to do
                continue
            subexpr_var = subexpr.owner.variables[identifier]
            if hasattr(subexpr_var, 'owner'):
                new_name = '_%s_%s_%s' % (name,
                                          subexpr.owner.name,
                                          identifier)
            else:
                new_name = '_%s_%s' % (name, identifier)
            substitutions[identifier] = new_name

            subexpr_var_index = group.variables.indices[identifier]
            if subexpr_var_index == group.variables.default_index:
                subexpr_var_index = index
            elif subexpr_var_index == '0':
                pass  # nothing to do for a shared variable
            elif index != self.default_index:
                index_var = self._variables.get(index, None)
                if isinstance(index_var, DynamicArrayVariable):
                    raise TypeError(('Cannot link to subexpression %s: it refers '
                                     'to the variable %s which is indexed with the '
                                     'dynamic index %s.') % (name,
                                                             identifier,
                                                             subexpr_var_index))
            else:
                self.add_reference(subexpr_var_index, group)

            self.indices[new_name] = subexpr_var_index

            if isinstance(subexpr_var, Subexpression):
                self.add_referred_subexpression(new_name,
                                                group,
                                                subexpr_var,
                                                subexpr_var_index)
            else:
                self.add_reference(new_name,
                                   group,
                                   identifier,
                                   subexpr_var_index)

        new_expr = word_substitute(subexpr.expr, substitutions)
        new_subexpr = Subexpression(name, subexpr.unit, self.owner, new_expr,
                                    device=subexpr.device,
                                    dtype=subexpr.dtype,
                                    scalar=subexpr.scalar)
        self._variables[name] = new_subexpr

    def add_reference(self, name, group, varname=None, index=None):
        '''
        Add a reference to a variable defined somewhere else (possibly under
        a different name). This is for example used in `Subgroup` and
        `Synapses` to refer to variables in the respective `NeuronGroup`.

        Parameters
        ----------
        name : str
            The name of the variable (in this group, possibly a different name
            from `var.name`).
        group : `Group`
            The group from which `var` is referenced
        varname : str, optional
            The variable to refer to. If not given, defaults to `name`.
        index : str, optional
            The index that should be used for this variable (defaults to
            `Variables.default_index`).
        '''
        if varname is None:
            varname = name
        if varname not in group.variables:
            raise KeyError(('Group {group} does not have a variable '
                            '{name}.').format(group=group.name,
                                              name=varname))
        if index is None:
            if group.variables[varname].scalar:
                index = '0'
            else:
                index = self.default_index

        if self.owner is not None and index in self.owner.variables:
            if (not self.owner.variables[index].read_only and
                    group.variables.indices[varname] != group.variables.default_index):
                raise TypeError(('Cannot link variable %s to %s in group %s -- '
                                 'need to precalculate direct indices but '
                                 'index %s can change') % (name,
                                                           varname,
                                                           group.name,
                                                           index))

        # We don't overwrite existing names with references
        if not name in self._variables:
            var = group.variables[varname]
            if isinstance(var, Subexpression):
                self.add_referred_subexpression(name, group, var, index)
            else:
                self._variables[name] = var
            self.indices[name] = index

    def add_references(self, group, varnames, index=None):
        '''
        Add all `Variable` objects from a name to `Variable` mapping with the
        same name as in the original mapping.

        Parameters
        ----------
        group : `Group`
            The group from which the `variables` are referenced
        varnames : iterable of str
            The variables that should be referred to in the current group
        index : str, optional
            The index to use for all the variables (defaults to
            `Variables.default_index`)
        '''
        for name in varnames:
            self.add_reference(name, group, name, index)

    def add_object(self, name, obj):
        '''
        Add an arbitrary Python object. This is only meant for internal use
        and therefore only names starting with an underscore are allowed.

        Parameters
        ----------
        name : str
            The name used for this object (has to start with an underscore).
        obj : object
            An arbitrary Python object that needs to be accessed directly from
            a `CodeObject`.
        '''
        if not name.startswith('_'):
            raise ValueError('This method is only meant for internally used '
                             'objects, the name therefore has to start with '
                             'an underscore')
        self._variables[name] = obj

    def create_clock_variables(self, clock, prefix=''):
        '''
        Convenience function to add the ``t`` and ``dt`` attributes of a
        `clock`.

        Parameters
        ----------
        clock : `Clock`
            The clock that should be used for ``t`` and ``dt``.
        prefix : str, optional
            A prefix for the variable names. Used for example in monitors to
            not confuse the dynamic array of recorded times with the current
            time in the recorded group.
        '''
        for name in ['t', 'dt']:
            self.add_reference(prefix+name, clock, name)
