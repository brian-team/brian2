'''
Classes used to specify the type of a function, variable or common
sub-expression.
'''
import weakref
import collections

import numpy as np

from brian2.utils.stringtools import get_identifiers
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
           'VariableView',
           'AuxiliaryVariable'
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


class Variable(object):
    '''
    An object providing information about model variables (including implicit
    variables such as ``t`` or ``xi``).
    
    Parameters
    ----------
    unit : `Unit`
        The unit of the variable. Note that the variable itself (as referenced
        by value) should never have units attached.
    owner : `Nameable`
        The object that "owns" this variable, e.g. the `NeuronGroup` or
        `Synapses` object that declares the variable in its model equations.
    name : 'str'
        The name of the variable. Note that this refers to the *original*
        name in the owning group. The same variable may be known under other
        names in other groups (e.g. the variable ``v`` of a `NeuronGroup` is
        known as ``v_post`` in a `Synapse` connecting to the group).
    dtype: `numpy.dtype`, optional
        The dtype used for storing the variable. Defaults to the preference
        `core.default_scalar.dtype` (or ``bool``, if `is_bool` is ``True``).
    scalar : bool, optional
        Whether the variable is a scalar value (``True``) or vector-valued, e.g.
        defined for every neuron (``False``). Defaults to ``True``.
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
    def __init__(self, unit, owner, name, dtype=None, scalar=False,
                 constant=False, is_bool=False, read_only=False):
        
        #: The variable's unit.
        self.unit = unit

        #: The variable's name.
        self.name = name

        try:
            owner = weakref.proxy(owner)
        except TypeError:  # Happens during testing with namedtuple
            pass
        #: The owner of the variable
        self.owner = owner

        if dtype is None:
            if is_bool:
                dtype = bool
            else:
                dtype = brian_prefs.core.default_scalar_dtype

        #: The dtype used for storing the variable.
        self.dtype = dtype

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
        owner_name = self.owner.name if not self.owner is None else 'None'
        description = ('<{classname}(unit={unit}, owner=<{owner}>,'
                       ' dtype={dtype}, scalar={scalar}, constant={constant},'
                       ' is_bool={is_bool}, read_only={read_only})>')
        return description.format(classname=self.__class__.__name__,
                                  unit=repr(self.unit),
                                  owner=owner_name,
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
    the `value`.

    Parameters
    ----------
    unit : `Unit`
        The unit of the variable. Note that the variable itself (as referenced
        by value) should never have units attached.
    value: reference to the variable value
        The value of the constant.
    '''
    def __init__(self, unit, value, owner, name):
        # Determine the type of the value
        dtype = get_dtype(value)
        is_bool = (value is True or value is False or
                   value is np.True_ or value is np.False_)

        #: The constant's value
        self.value = value

        super(Constant, self).__init__(unit=unit, owner=owner, name=name,
                                       dtype=dtype, scalar=True, constant=True,
                                       read_only=True, is_bool=is_bool)

    def get_value(self):
        return self.value

class AuxiliaryVariable(Variable):
    '''
    Variable description for an auxiliary variable (most likely one that is
    added automatically to abstract code, e.g. ``_cond`` for a threshold
    condition), specifying its type and unit for code generation.

    Parameters
    ----------
    unit : `Unit`
        The unit of the variable.
    dtype: `numpy.dtype`, optional
        The dtype used for storing the variable. If none is given, defaults
        to `core.default_scalar_dtype`.
    scalar : bool, optional
        Whether the variable is a scalar value (``True``) or vector-valued, e.g.
        defined for every neuron (``False``). Defaults to ``False``.
    is_bool: bool, optional
        Whether this is a boolean variable (also implies it is dimensionless).
        Defaults to ``False``.
    '''
    def __init__(self, unit, name, dtype=None, scalar=False, is_bool=False):
        if dtype is None:
            if is_bool:
                dtype = bool
            else:
                dtype = brian_prefs.core.default_scalar_dtype
        super(AuxiliaryVariable, self).__init__(unit=unit, owner=None,
                                                name=name, dtype=dtype,
                                                scalar=scalar, is_bool=is_bool)


class AttributeVariable(Variable):
    '''
    An object providing information about a value saved as an attribute of an
    object. Instead of saving a reference to the value itself, we save the
    name of the attribute. This way, we get the correct value if the attribute
    is overwritten with a new value (e.g. in the case of ``clock.t_``)
    
    The object value has to be accessible by doing ``getattr(obj, attribute)``.
    
    Parameters
    ----------
    unit : `Unit`
        The unit of the variable
    owner : `Nameable`
        The object that "owns" this variable, e.g. the `NeuronGroup` or
        `Synapses` object that declares the variable in its model equations.
    attribute : str
        The name of the attribute storing the variable's value. `attribute` has
        to be an attribute of `owner`.
    constant : bool, optional
        Whether the attribute's value is constant during a run. Defaults to
        ``False``.
    read_only : bool, optional
        Whether this is a read-only variable, i.e. a variable that is set
        internally and cannot be changed by the user (this is used for example
        for the variable ``N``, the number of neurons in a group). Defaults
        to ``False``.
    scalar : bool, optional
        Whether the variable is a scalar value (``True``) or vector-valued, e.g.
        defined for every neuron (``False``). Defaults to ``True``.
    is_bool: bool, optional
        Whether this is a boolean variable (also implies it is dimensionless).
        Defaults to ``False``.
    '''
    def __init__(self, unit, owner, name, attribute, dtype, constant=False, scalar=True,
                 is_bool=False):
        super(AttributeVariable, self).__init__(unit=unit, owner=owner,
                                                name=name, dtype=dtype,
                                                constant=constant,
                                                scalar=scalar,
                                                is_bool=is_bool, read_only=True)

        #: The name of the attribute storing the variable's value
        self.attribute = attribute

    def get_value(self):
        return getattr(self.owner, self.attribute)

    def __repr__(self):
        owner_name = self.owner.name if not self.owner is None else 'None'
        description = ('{classname}(unit={unit}, owner=<{owner}>, '
                       'attribute={attribute}, constant={constant})')
        return description.format(classname=self.__class__.__name__,
                                  unit=repr(self.unit),
                                  owner=owner_name,
                                  attribute=repr(self.attribute),
                                  constant=repr(self.constant))


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
         is accessed without units (e.g. when stating `G.var_`).
    '''

    def __init__(self, name, variable, group, unit=None):
        self.name = name
        self.variable = variable
        self.group = weakref.proxy(group)
        self.unit = unit

    def __getitem__(self, item):
        variable = self.variable
        if isinstance(item, basestring):
            values = variable.device.get_with_expression(self.group, self.name,
                                                         variable, item,
                                                         level=1)
        else:
            values = variable.device.get_with_index_array(self.group,
                                                          self.name, variable,
                                                          item)

        if self.unit is None:
            return values
        else:
            return Quantity(values, self.unit.dimensions)

    def __setitem__(self, item, value):
        variable = self.variable
        if variable.read_only:
            raise TypeError('Variable %s is read-only.' % self.name)

        if isinstance(item, slice) and item == slice(None):
            item = 'True'

        check_units = self.unit is not None

        # Both index and values are strings, use a single code object do deal
        # with this situation
        if isinstance(value, basestring) and isinstance(item, basestring):
            variable.device.set_with_expression_conditional(self.group, self.name,
                                                            item, value,
                                                            check_units=check_units,
                                                            level=1)
        elif isinstance(item, basestring):
            try:
                float(value)  # only checks for the exception
            except (TypeError, ValueError):
                if item == 'True':
                    # Fall back to the general array-array pattern
                    variable.device.set_with_index_array(self.group, self.name,
                                                         self.variable,
                                                         slice(None), value,
                                                         check_units=check_units)
                    return
                else:
                    raise TypeError('When setting a variable based on a string '
                                    'index, the value has to be a string or a '
                                    'scalar.')

            variable.device.set_with_expression_conditional(self.group, self.name,
                                                            item, repr(value),
                                                            check_units=check_units,
                                                            level=1)
        elif isinstance(value, basestring):
            variable.device.set_with_expression(self.group, self.name, item, value,
                                                check_units=check_units,
                                                level=1)
        else:  # No string expressions involved
            variable.device.set_with_index_array(self.group, self.name,
                                                 self.variable, item, value,
                                                 check_units=check_units)

    # Allow some basic calculations directly on the ArrayView object

    def __array__(self, dtype=None):
        if dtype is not None and dtype != self.variable.dtype:
            raise NotImplementedError('Changing dtype not supported')
        return self[:]

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

    def __repr__(self):
        varname = self.name
        if self.unit is None:
            varname += '_'
        return '<%s.%s: %r>' % (self.group.name, varname,
                                 self[:])


class ArrayVariable(Variable):
    '''
    An object providing information about a model variable stored in an array
    (for example, all state variables).

    Parameters
    ----------
    unit : `Unit`
        The unit of the variable
    owner : `Nameable`
        The object that "owns" this variable, e.g. the `NeuronGroup` or
        `Synapses` object that declares the variable in its model equations.
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
    def __init__(self, unit, owner, name, size, device, dtype=None,
                 constant=False, scalar=False, is_bool=False, read_only=False):
        super(ArrayVariable, self).__init__(unit=unit, owner=owner, name=name,
                                            dtype=dtype, scalar=scalar,
                                            constant=constant, is_bool=is_bool,
                                            read_only=read_only)
        self.device = device
        self.size = size

    def get_value(self):
        return self.device.get_value(self)

    def set_value(self, value):
        self.device.fill_with_array(self, value)

    def get_size(self):
        return self.size

    def get_addressable_value(self, name, group):
        return VariableView(name=name, variable=self, group=group, unit=None)

    def get_addressable_value_with_unit(self, name, group):
        return VariableView(name=name, variable=self, group=group,
                            unit=self.unit)


class DynamicArrayVariable(ArrayVariable):
    '''
    An object providing information about a model variable stored in a dynamic
    array (used in `Synapses`).

    Parameters
    ----------
    unit : `Unit`
        The unit of the variable
    value : `numpy.ndarray`
        A reference to the array storing the data for the variable.
    owner : `Nameable`
        The object that "owns" this variable, e.g. the `NeuronGroup` or
        `Synapses` object that declares the variable in its model equations.
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

    def __init__(self, unit, owner, name, size, device, dtype=None,
                 constant=False, constant_size=True,
                 scalar=False, is_bool=False, read_only=False):
        #: The number of dimensions
        if isinstance(size, int):
            self.dimensions = 1
        else:
            self.dimensions = len(size)

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
        self.device.resize(self, new_size)
        self.size = new_size


class Subexpression(Variable):
    '''
    An object providing information about a named subexpression in a model.
    
    Parameters
    ----------
    name : str
        The name of the subexpression.
    unit : `Unit`
        The unit of the subexpression.
    expr : str
        The subexpression itself.
    owner : `Group`
        The group to which the expression refers.
    dtype : `numpy.dtype`, optional
        The dtype used for the expression. Defaults to
        `core.default_scalar_dtype`.
    is_bool: bool, optional
        Whether this is a boolean variable (also implies it is dimensionless).
        Defaults to ``False``
    '''
    def __init__(self, unit, expr, owner, name, device, dtype=None, is_bool=False):
        super(Subexpression, self).__init__(unit=unit, owner=owner,
                                            name=name, dtype=dtype,
                                            is_bool=is_bool, scalar=False,
                                            constant=False, read_only=True)

        self.device = device

        #: The expression defining the subexpression
        self.expr = expr.strip()
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
        owner_name = self.owner.name if not self.owner is None else 'None'
        description = ('<{classname}(name={name}, unit={unit}, dtype={dtype}, '
                       'expr={expr}, owner=<{owner}>, is_bool={is_bool})>')
        return description.format(classname=self.__class__.__name__,
                                  name=repr(self.name),
                                  unit=repr(self.unit),
                                  dtype=repr(self.dtype),
                                  expr=repr(self.expr),
                                  owner=owner_name,
                                  is_bool=repr(self.is_bool))


class Variables(collections.Mapping):
    '''
    A container class for storing `Variable` objects. Instances of this class
    are used as the `Group.variables` attribute and can be accessed as
    (read-only) dictionaries.
    '''

    @staticmethod
    def get_dtype(dtype, is_bool=False):
        if is_bool:
            return np.bool
        elif dtype is None:
            return brian_prefs.core.default_scalar_dtype
        else:
            return dtype

    def __init__(self, owner, default_index='_idx'):
        #: A reference to the `Group` owning these variables
        self.owner = owner
        # The index that is used for arrays if no index is given explicitly
        self.default_index = default_index

        # We do the import here to avoid a circular dependency.
        from brian2.devices.device import get_device
        self.device = get_device()

        self._variables = {}
        #: A dictionary given the index name for every array name
        self.indices = collections.defaultdict(lambda: default_index)

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

    def add_array(self, name, size, unit, dtype=None,
                  constant=False, is_bool=False, read_only=False,
                  index=None):
        var = ArrayVariable(unit, owner=self.owner, name=name, device=self.device,
                            size=size, dtype=Variables.get_dtype(dtype, is_bool),
                            constant=constant, is_bool=is_bool,
                            read_only=read_only)
        self._add_variable(name, var, index)
        self.device.init_with_zeros(var)

    def add_dynamic_array(self, name, size, unit, dtype=None, constant=False,
                          constant_size=False, is_bool=False, read_only=False,
                          index=None):
        var = DynamicArrayVariable(unit, owner=self.owner, name=name, device=self.device,
                                   size=size, dtype=Variables.get_dtype(dtype, is_bool),
                                   constant=constant, constant_size=constant_size,
                                   is_bool=is_bool, read_only=read_only)
        self._add_variable(name, var, index)

    def add_arange(self, name, size, start=0, dtype=np.int32, constant=True,
                   read_only=True, index=None):
        self.add_array(name, size, unit=Unit(1), dtype=dtype, constant=constant,
                       is_bool=False, read_only=read_only, index=index)
        self.device.init_with_arange(self._variables[name], start)

    def add_attribute_variable(self, name, unit, owner, attribute, dtype=None,
                               constant=False, scalar=True):
        if dtype is None:
            value = getattr(owner, attribute, None)
            if value is None:
                raise ValueError(('Cannot determine dtype for attribute "%s" '
                                  'of object "%r"') % (attribute, owner))
            dtype = get_dtype(value)

        var = AttributeVariable(unit, owner=owner, name=name, attribute=attribute,
                                dtype=dtype, constant=constant, scalar=scalar)
        self._add_variable(name, var)

    def add_constant(self, name, unit, value):
        var = Constant(unit, owner=self.owner, name=name, value=value)
        self._add_variable(name, var)

    def add_subexpression(self, name, unit, expr, dtype=None, is_bool=False):
        var = Subexpression(name=name, unit=unit, expr=expr, owner=self.owner,
                            dtype=dtype, device=self.device, is_bool=is_bool)
        self._add_variable(name, var)

    def add_auxiliary_variable(self, name, unit, dtype=None, scalar=False,
                               is_bool=False):
        var = AuxiliaryVariable(unit, name=name, dtype=dtype, scalar=scalar,
                                is_bool=is_bool)
        self._add_variable(name, var)

    def add_reference(self, name, var, index=None):
        if index is None:
            index = self.default_index
        # We don't overwrite existing names with references
        if not name in self._variables:
            self._variables[name] = var
            self.indices[name] = index

    def add_references(self, variables, index=None):
        '''
        Add all `Variable` objects from a name to `Variable` mapping.
        '''
        for name, var in variables.iteritems():
            self.add_reference(name, var, index)

    def add_clock_variables(self, clock):
        '''
        Convenience function to add the ``t`` and ``dt`` attributes of a
        `clock`.

        Parameters
        ----------
        clock : `Clock`
            The clock that should be used for ``t`` and ``dt``. Note that the
            actual attributes referred to are ``t_`` and ``dt_``, i.e. the
            unitless values.
        '''
        self.add_attribute_variable('t', unit=second, owner=clock,
                                    attribute='t_', dtype=np.float64)
        self.add_attribute_variable('dt', unit=second, owner=clock,
                                    attribute='dt_', dtype=np.float64,
                                    constant=True)