'''
Classes used to specify the type of a function, variable or common
sub-expression.
'''
import collections
import functools

import sympy
import numpy as np

from brian2.core.base import weakproxy_with_fallback
from brian2.utils.stringtools import get_identifiers, word_substitute
from brian2.units.fundamentalunits import (Quantity, Unit,
                                           fail_for_dimension_mismatch,
                                           have_same_dimensions)
from brian2.units.allunits import second

from .preferences import brian_prefs

__all__ = ['Variable',
           'Constant',
           'AttributeVariable',
           'ArrayVariable',
           'DynamicArrayVariable',
           'Subexpression',
           'AuxiliaryVariable',
           'VariableView',
           'Variables'
           ]


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


def default_dtype(dtype, is_bool=False):
    '''
    Helper function to return the default dtype.

    Parameters
    ----------
    dtype : `dtype` or ``None``
        The dtype to use (or `None` to use the default)
    is_bool : bool
        If `is_bool` is ``True``, ``bool`` is used as the dtype.

    Returns
    -------
    final_dtype : `dtype`
    '''
    if is_bool:
        return np.bool
    elif dtype is None:
        return brian_prefs.core.default_scalar_dtype
    else:
        return dtype


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
    dtype : `dtype`, optional
        The dtype used for storing the variable. Defaults to the preference
        `core.default_scalar.dtype` (or ``bool``, if `is_bool` is ``True``).
    scalar : bool, optional
        Whether the variable is a scalar value (``True``) or vector-valued, e.g.
        defined for every neuron (``False``). Defaults to ``False``.
    constant: bool, optional
        Whether the value of this variable can change during a run. Defaults
        to ``False``.
    is_bool: bool, optional
        Whether this is a boolean variable (also implies it is dimensionless).
        Defaults to ``False``.
    read_only : bool, optional
        Whether this is a read-only variable, i.e. a variable that is set
        internally and cannot be changed by the user (this is used for example
        for the variable ``N``, the number of neurons in a group). Defaults
        to ``False``.
    '''
    def __init__(self, name, unit, dtype=None, scalar=False,
                 constant=False, is_bool=False, read_only=False):
        
        #: The variable's unit.
        self.unit = unit

        #: The variable's name.
        self.name = name

        #: The dtype used for storing the variable.
        self.dtype = default_dtype(dtype, is_bool)

        #: Whether this variable is a boolean
        self.is_bool = is_bool

        if is_bool:
            if not have_same_dimensions(unit, 1):
                raise ValueError('Boolean variables can only be dimensionless')

        #: Whether the variable is a scalar
        self.scalar = scalar

        #: Whether the variable is constant during a run
        self.constant = constant

        #: Whether the variable is read-only
        self.read_only = read_only

    @property
    def dim(self):
        '''
        The dimensions of this variable.
        '''
        return self.unit.dim

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
                       ' is_bool={is_bool}, read_only={read_only})>')
        return description.format(classname=self.__class__.__name__,
                                  unit=repr(self.unit),
                                  dtype=repr(self.dtype),
                                  scalar=repr(self.scalar),
                                  constant=repr(self.constant),
                                  is_bool=repr(self.is_bool),
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
    '''
    def __init__(self, name, unit, value):
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

        super(Constant, self).__init__(unit=unit, name=name,
                                       dtype=dtype, scalar=True, constant=True,
                                       read_only=True, is_bool=is_bool)

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
        to `core.default_scalar_dtype`.
    scalar : bool, optional
        Whether the variable is a scalar value (``True``) or vector-valued, e.g.
        defined for every neuron (``False``). Defaults to ``False``.
    is_bool: bool, optional
        Whether this is a boolean variable (also implies it is dimensionless).
        Defaults to ``False``.
    '''
    def __init__(self, name, unit, dtype=None, scalar=False, is_bool=False):
        super(AuxiliaryVariable, self).__init__(unit=unit,
                                                name=name, dtype=dtype,
                                                scalar=scalar, is_bool=is_bool)

    def get_value(self):
        raise TypeError('Cannot get the value for an auxiliary variable.')


class AttributeVariable(Variable):
    '''
    An object providing information about a value saved as an attribute of an
    object. Instead of saving a reference to the value itself, we save the
    name of the attribute. This way, we get the correct value if the attribute
    is overwritten with a new value (e.g. in the case of ``clock.t_``).
    Most of the time `Variables.add_attribute_variable` should be used instead
    of instantiating this class directly.
    
    The object value has to be accessible by doing ``getattr(obj, attribute)``.
    Variables of this type are considered read-only.
    
    Parameters
    ----------
    name : str
        The name of the variable
    unit : `Unit`
        The unit of the variable
    obj : object
        The object storing the attribute.
    attribute : str
        The name of the attribute storing the variable's value. `attribute` has
        to be an attribute of `obj`.
    dtype : `dtype`, optional
        The dtype used for storing the variable. If none is given, defaults
        to `core.default_scalar_dtype`.
    constant : bool, optional
        Whether the attribute's value is constant during a run. Defaults to
        ``False``.
    scalar : bool, optional
        Whether the variable is a scalar value (``True``) or vector-valued, e.g.
        defined for every neuron (``False``). Defaults to ``True``.
    is_bool: bool, optional
        Whether this is a boolean variable (also implies it is dimensionless).
        Defaults to ``False``.
    '''
    def __init__(self, name, unit, obj, attribute, dtype, constant=False,
                 scalar=True, is_bool=False):
        super(AttributeVariable, self).__init__(unit=unit,
                                                name=name, dtype=dtype,
                                                constant=constant,
                                                scalar=scalar,
                                                is_bool=is_bool, read_only=True)

        #: The object storing the `attribute`
        self.obj = obj

        #: The name of the attribute storing the variable's value
        self.attribute = attribute

    def get_value(self):
        return getattr(self.obj, self.attribute)

    def __repr__(self):
        description = ('{classname}(unit={unit}, obj=<{obj}>, '
                       'attribute={attribute}, constant={constant})')
        return description.format(classname=self.__class__.__name__,
                                  unit=repr(self.unit),
                                  obj=self.obj,
                                  attribute=repr(self.attribute),
                                  constant=repr(self.constant))


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
        to `core.default_scalar_dtype`.
    constant : bool, optional
        Whether the variable's value is constant during a run.
        Defaults to ``False``.
    scalar : bool, optional
        Whether this array is a 1-element array that should be treated like a
        scalar (e.g. for a single delay value across synapses). Defaults to
        ``False``.
    is_bool: bool, optional
        Whether this is a boolean variable (also implies it is dimensionless).
        Defaults to ``False``
    read_only : bool, optional
        Whether this is a read-only variable, i.e. a variable that is set
        internally and cannot be changed by the user. Defaults
        to ``False``.
    '''
    def __init__(self, name, unit, owner, size, device, dtype=None,
                 constant=False, scalar=False, is_bool=False, read_only=False):
        super(ArrayVariable, self).__init__(unit=unit, name=name,
                                            dtype=dtype, scalar=scalar,
                                            constant=constant, is_bool=is_bool,
                                            read_only=read_only)
        #: The `Group` to which this variable belongs.
        self.owner = weakproxy_with_fallback(owner)

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
        if not var.is_bool:
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
        to `core.default_scalar_dtype`.
    constant : bool, optional
        Whether the variable's value is constant during a run.
        Defaults to ``False``.
    constant_size : bool, optional
        Whether the size of the variable is constant during a run.
        Defaults to ``True``.
    scalar : bool, optional
        Whether this array is a 1-element array that should be treated like a
        scalar (e.g. for a single delay value across synapses). Defaults to
        ``False``.
    is_bool: bool, optional
        Whether this is a boolean variable (also implies it is dimensionless).
        Defaults to ``False``
    read_only : bool, optional
        Whether this is a read-only variable, i.e. a variable that is set
        internally and cannot be changed by the user. Defaults
        to ``False``.
    '''

    def __init__(self, name, unit, owner, size, device, dtype=None,
                 constant=False, constant_size=True,
                 scalar=False, is_bool=False, read_only=False):

        if isinstance(size, int):
            dimensions = 1
        else:
            dimensions = len(size)

        #: The number of dimensions
        self.dimensions = dimensions

        if constant and not constant_size:
            raise ValueError('A variable cannot be constant and change in size')
        #: Whether the size of the variable is constant during a run.
        self.constant_size = constant_size
        super(DynamicArrayVariable, self).__init__(unit=unit,
                                                   owner=owner,
                                                   name=name,
                                                   size=size,
                                                   device=device,
                                                   constant=constant,
                                                   dtype=dtype,
                                                   scalar=scalar,
                                                   is_bool=is_bool,
                                                   read_only=read_only)
    def resize(self, new_size):
        '''
        Resize the dynamic array. Calls `self.device.resize` to do the
        actual resizing.

        Parameters
        ----------
        new_size : int or tuple of int
            The new size.
        '''
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
        `core.default_scalar_dtype`.
    is_bool: bool, optional
        Whether this is a boolean variable (also implies it is dimensionless).
        Defaults to ``False``
    scalar: bool, optional
        Whether this is an expression only referring to scalar variables.
        Defaults to ``False``
    '''
    def __init__(self, name, unit, owner, expr, device, dtype=None,
                 is_bool=False, scalar=False):
        super(Subexpression, self).__init__(unit=unit,
                                            name=name, dtype=dtype,
                                            is_bool=is_bool, scalar=scalar,
                                            constant=False, read_only=True)
        #: The `Group` to which this variable belongs
        self.owner = weakproxy_with_fallback(owner)

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
                       'expr={expr}, owner=<{owner}>, is_bool={is_bool})>')
        return description.format(classname=self.__class__.__name__,
                                  name=repr(self.name),
                                  unit=repr(self.unit),
                                  dtype=repr(self.dtype),
                                  expr=repr(self.expr),
                                  owner=self.owner.name,
                                  is_bool=repr(self.is_bool))


# ------------------------------------------------------------------------------
# Classes providing views on variables and storing variables information
# ------------------------------------------------------------------------------
class VariableView(object):
    '''
    A view on a variable that allows to treat it as an numpy array while
    allowing special indexing (e.g. with strings) in the context of a `Group`.

    Parameters
    ----------
    name : str
        The name of the variable
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
        self.group = weakproxy_with_fallback(group)
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
        variable = self.variable
        if isinstance(item, basestring):
            values = self.group.get_with_expression(self.name,
                                                    variable, item,
                                                    level=level+1,
                                                    run_namespace=namespace)
        else:
            values = self.group.get_with_index_array(self.name, variable,
                                                     item)

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

        if isinstance(item, slice) and item == slice(None):
            item = 'True'

        check_units = self.unit is not None

        # Both index and values are strings, use a single code object do deal
        # with this situation
        if isinstance(value, basestring) and isinstance(item, basestring):
            self.group.set_with_expression_conditional(self.name,
                                                       variable,
                                                       item, value,
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
                    self.group.set_with_index_array(self.name,
                                                    variable,
                                                    slice(None), value,
                                                    check_units=check_units)
            else:
                self.group.set_with_expression_conditional(self.name,
                                                           variable,
                                                           item,
                                                           repr(value),
                                                           check_units=check_units,
                                                           level=level+1,
                                                           run_namespace=namespace)
        elif isinstance(value, basestring):
            self.group.set_with_expression(self.name, variable,
                                           item, value,
                                           check_units=check_units,
                                           level=level+1,
                                           run_namespace=namespace)
        else:  # No string expressions involved
            self.group.set_with_index_array(self.name,
                                            variable, item, value,
                                            check_units=check_units)

    def __setitem__(self, item, value):
        self.set_item(item, value, level=1)

    # Allow some basic calculations directly on the ArrayView object

    def __array__(self, dtype=None):
        if dtype is not None and dtype != self.variable.dtype:
            raise NotImplementedError('Changing dtype not supported')
        return self[:]

    def __len__(self):
        return len(self[:])

    def __neg__(self):
        return -self[:]

    def __pos__(self):
        return self[:]

    def __add__(self, other):
        return self[:] + other

    def __radd__(self, other):
        return other + self[:]

    def __sub__(self, other):
        return self[:] - other

    def __rsub__(self, other):
        return other - self[:]

    def __mul__(self, other):
        return self[:] * other

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        return self[:] / other

    def __truediv__(self, other):
        return self[:] / other

    def __rdiv__(self, other):
        return other / self[:]

    def __rtruediv__(self, other):
        return other / self[:]

    def __iadd__(self, other):
        if isinstance(other, basestring):
            raise TypeError(('In-place modification with strings not '
                             'supported. Use group.var = "var + expression" '
                             'instead of group.var += "expression".'))
        else:
            rhs = self[:] + other
        self[:] = rhs
        return self

    def __isub__(self, other):
        if isinstance(other, basestring):
            raise TypeError(('In-place modification with strings not '
                             'supported. Use group.var = "var - expression" '
                             'instead of group.var -= "expression".'))
        else:
            rhs = self[:] - other
        self[:] = rhs
        return self

    def __imul__(self, other):
        if isinstance(other, basestring):
            raise TypeError(('In-place modification with strings not '
                             'supported. Use group.var = "var * expression" '
                             'instead of group.var *= "expression".'))
        else:
            rhs = self[:] * other
        self[:] = rhs
        return self

    def __idiv__(self, other):
        if isinstance(other, basestring):
            raise TypeError(('In-place modification with strings not '
                             'supported. Use group.var = "var / expression" '
                             'instead of group.var /= "expression".'))
        else:
            rhs = self[:] / other
        self[:] = rhs
        return self

    # Also allow logical comparisons

    def __eq__(self, other):
        return self[:] == other

    def __ne__(self, other):
        return self[:] != other

    def __lt__(self, other):
        return self[:] < other

    def __le__(self, other):
        return self[:] <= other

    def __gt__(self, other):
        return self[:] > other

    def __ge__(self, other):
        return self[:] >= other

    def __repr__(self):
        varname = self.name
        if self.unit is None:
            varname += '_'
        return '<%s.%s: %r>' % (self.group.name, varname,
                                 self[:])


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

        if index is not None:
            self.indices[name] = index

    def add_array(self, name, unit, size, dtype=None,
                  constant=False, is_bool=False, read_only=False, scalar=False,
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
        dtype : `dtype`, optional
            The dtype used for storing the variable. If none is given, defaults
            to `core.default_scalar_dtype`.
        constant : bool, optional
            Whether the variable's value is constant during a run.
            Defaults to ``False``.
        scalar : bool, optional
            Whether this is a scalar variable. Defaults to ``False``, if set to
            ``True``, also implies that `size` equals 1.
        is_bool: bool, optional
            Whether this is a boolean variable (also implies it is
            dimensionless). Defaults to ``False``
        read_only : bool, optional
            Whether this is a read-only variable, i.e. a variable that is set
            internally and cannot be changed by the user. Defaults
            to ``False``.
        index : str, optional
            The index to use for this variable. Defaults to
            `Variables.default_index`.
        '''
        var = ArrayVariable(name=name, unit=unit, owner=self.owner,
                            device=self.device, size=size,
                            dtype=default_dtype(dtype, is_bool),
                            constant=constant, is_bool=is_bool,
                            scalar=scalar,
                            read_only=read_only)
        self._add_variable(name, var, index)
        self.device.init_with_zeros(var)

    def add_dynamic_array(self, name, unit, size, dtype=None, constant=False,
                          constant_size=True, is_bool=False, read_only=False,
                          index=None):
        '''
        Add a dynamic array.

        Parameters
        ----------
        name : str
            The name of the variable.
        unit : `Unit`
            The unit of the variable
        size : int or tuple of int
            The size of the array.
        dtype : `dtype`, optional
            The dtype used for storing the variable. If none is given, defaults
            to `core.default_scalar_dtype`.
        constant : bool, optional
            Whether the variable's value is constant during a run.
            Defaults to ``False``.
        constant_size : bool, optional
            Whether the size of the variable is constant during a run.
            Defaults to ``True``.
        is_bool: bool, optional
            Whether this is a boolean variable (also implies it is
            dimensionless). Defaults to ``False``
        read_only : bool, optional
            Whether this is a read-only variable, i.e. a variable that is set
            internally and cannot be changed by the user. Defaults
            to ``False``.
        index : str, optional
            The index to use for this variable. Defaults to
            `Variables.default_index`.
        '''
        var = DynamicArrayVariable(name=name, unit=unit, owner=self.owner,
                                   device=self.device,
                                   size=size, dtype=default_dtype(dtype, is_bool),
                                   constant=constant, constant_size=constant_size,
                                   is_bool=is_bool, read_only=read_only)
        self._add_variable(name, var, index)

    def add_arange(self, name, size, start=0, dtype=np.int32, constant=True,
                   read_only=True, index=None):
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
        '''
        self.add_array(name=name, unit=Unit(1), size=size, dtype=dtype,
                       constant=constant, is_bool=False, read_only=read_only,
                       index=index)
        self.device.init_with_arange(self._variables[name], start)

    def add_attribute_variable(self, name, unit, obj, attribute, dtype=None,
                               constant=False, scalar=True, is_bool=False):
        '''
        Add a variable stored as an attribute of an object.

        Parameters
        ----------
        name : str
            The name of the variable
        unit : `Unit`
            The unit of the variable
        obj : object
            The object storing the attribute.
        attribute : str
            The name of the attribute storing the variable's value. `attribute` has
            to be an attribute of `obj`.
        dtype : `dtype`, optional
            The dtype used for storing the variable. If none is given, uses the
            type of ``obj.attribute`` (which have to exist).
        constant : bool, optional
            Whether the attribute's value is constant during a run. Defaults to
            ``False``.
        scalar : bool, optional
            Whether the variable is a scalar value (``True``) or vector-valued, e.g.
            defined for every neuron (``False``). Defaults to ``True``.
        is_bool: bool, optional
            Whether this is a boolean variable (also implies it is dimensionless).
            Defaults to ``False``.
        '''
        if dtype is None:
            value = getattr(obj, attribute, None)
            if value is None:
                raise ValueError(('Cannot determine dtype for attribute "%s" '
                                  'of object "%r"') % (attribute, obj))
            dtype = get_dtype(value)

        var = AttributeVariable(name=name, unit=unit, obj=obj,
                                attribute=attribute, dtype=dtype,
                                constant=constant, scalar=scalar,
                                is_bool=is_bool)
        self._add_variable(name, var)

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
        var = Constant(name=name, unit=unit, value=value)
        self._add_variable(name, var)

    def add_subexpression(self, name, unit, expr, dtype=None, is_bool=False,
                          scalar=False):
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
            `core.default_scalar_dtype`.
        is_bool : bool, optional
            Whether this is a boolean variable (also implies it is
            dimensionless). Defaults to ``False``
        scalar : bool, optional
            Whether this is an expression only referring to scalar variables.
            Defaults to ``False``
        '''
        var = Subexpression(name=name, unit=unit, expr=expr, owner=self.owner,
                            dtype=dtype, device=self.device, is_bool=is_bool,
                            scalar=scalar)
        self._add_variable(name, var)

    def add_auxiliary_variable(self, name, unit, dtype=None, scalar=False,
                               is_bool=False):
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
            to `core.default_scalar_dtype`.
        scalar : bool, optional
            Whether the variable is a scalar value (``True``) or vector-valued,
            e.g. defined for every neuron (``False``). Defaults to ``False``.
        is_bool: bool, optional
            Whether this is a boolean variable (also implies it is
            dimensionless). Defaults to ``False``.
        '''
        var = AuxiliaryVariable(name=name, unit=unit, dtype=dtype,
                                scalar=scalar, is_bool=is_bool)
        self._add_variable(name, var)


    def add_referred_subexpression(self, name, subexpr, index):
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
            self.indices[new_name] = index
            if isinstance(subexpr_var, Subexpression):
                self.add_referred_subexpression(new_name, subexpr_var, index)
            else:
                self.add_reference(new_name, subexpr_var, index)
        new_expr = word_substitute(subexpr.expr, substitutions)
        new_subexpr = Subexpression(name, subexpr.unit, self.owner, new_expr,
                                    device=subexpr.device,
                                    dtype=subexpr.dtype,
                                    is_bool=subexpr.is_bool,
                                    scalar=subexpr.scalar)
        self._variables[name] = new_subexpr

    def add_reference(self, name, var, index=None):
        '''
        Add a reference to a variable defined somewhere else (possibly under
        a different name). This is for example used in `Subgroup` and
        `Synapses` to refer to variables in the respective `NeuronGroup`.

        Parameters
        ----------
        name : str
            The name of the variable (in this group, possibly a different name
            from `var.name`).
        var : `Variable`
            The variable to refer to.
        index : str, optional
            The index that should be used for this variable (defaults to
            `Variables.default_index`).
        '''
        if index is None:
            index = self.default_index
        # We don't overwrite existing names with references
        if not name in self._variables:
            if isinstance(var, Subexpression):
                self.add_referred_subexpression(name, var, index)
            else:
                self._variables[name] = var
            self.indices[name] = index

    def add_references(self, variables, index=None):
        '''
        Add all `Variable` objects from a name to `Variable` mapping with the
        same name as in the original mapping.

        Parameters
        ----------
        variables : mapping from str to `Variable` (normally a `Variables` object)
            The variables that should be referred to in the current group
        index : str, optional
            The index to use for all the variables (defaults to
            `Variables.default_index`)
        '''
        for name, var in variables.iteritems():
            self.add_reference(name, var, index)

    def add_clock_variables(self, clock, prefix=''):
        '''
        Convenience function to add the ``t`` and ``dt`` attributes of a
        `clock`.

        Parameters
        ----------
        clock : `Clock`
            The clock that should be used for ``t`` and ``dt``. Note that the
            actual attributes referred to are ``t_`` and ``dt_``, i.e. the
            unitless values.
        prefix : str, optional
            A prefix for the variable names. Used for example in monitors to
            not confuse the dynamic array of recorded times with the current
            time in the recorded group.
        '''
        self.add_attribute_variable(prefix+'t', unit=second, obj=clock,
                                    attribute='t_', dtype=np.float64)
        self.add_attribute_variable(prefix+'dt', unit=second, obj=clock,
                                    attribute='dt_', dtype=np.float64,
                                    constant=True)
