'''
Classes used to specify the type of a function, variable or common
sub-expression.
'''
import weakref

import numpy as np

from brian2.units.allunits import second

from brian2.utils.stringtools import get_identifiers
from brian2.units.fundamentalunits import (Quantity, Unit, is_scalar_type,
                                           fail_for_dimension_mismatch,
                                           have_same_dimensions)

__all__ = ['Variable',
           'StochasticVariable',
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
    value: reference to the variable value, optional
        Some variables (e.g. stochastic variables) don't have their value
        stored anywhere, they'd pass ``None`` as a value.
    dtype: `numpy.dtype`, optional
        The dtype used for storing the variable. If none is given, tries to
        get the dtype from the referred value.
    scalar : bool, optional
        Whether the variable is a scalar value (``True``) or vector-valued, e.g.
        defined for every neuron (``False``). If nothing is specified,
        determines the correct setting from the `value`, if that is not given
        defaults to ``True``.
    constant: bool, optional
        Whether the value of this variable can change during a run. Defaults
        to ``False``.
    is_bool: bool, optional
        Whether this is a boolean variable (also implies it is dimensionless).
        If specified as ``None`` and a `value` is given, checks the value
        itself. If no `value` is given, defaults to ``False``.
    read_only : bool, optional
        Whether this is a read-only variable, i.e. a variable that is set
        internally and cannot be changed by the user (this is used for example
        for the variable ``N``, the number of neurons in a group). Defaults
        to ``False``.
    '''
    def __init__(self, unit, value=None, dtype=None, scalar=None,
                 constant=False, is_bool=None, read_only=False):
        
        #: The variable's unit.
        self.unit = unit

        #: reference to a value of type `dtype`
        self.value = value

        if dtype is None:
            self.dtype = get_dtype(value)
        else:
            value_dtype = get_dtype(value)
            if value is not None and value_dtype != dtype:
                raise TypeError(('Conflicting dtype information: '
                                 'referred value has dtype %r, not '
                                 '%r.') % (value_dtype, dtype))
            #: The dtype used for storing the variable.
            self.dtype = dtype

        if is_bool is None:
            if value is None:
                self.is_bool = False
            self.is_bool = value is True or value is False
        else:
            #: Whether this variable is a boolean
            self.is_bool = is_bool

        if is_bool:
            if not have_same_dimensions(unit, 1):
                raise ValueError('Boolean variables can only be dimensionless')

        if scalar is None:
            if value is None:
                self.scalar = True
            self.scalar = is_scalar_type(value)
        else:
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
        if self.value is None:
            raise TypeError('Variable does not have a value')
        else:
            return self.value

    def set_value(self, value, index=None):
        '''
        Set the value associated with the variable.
        '''
        raise NotImplementedError()

    def get_value_with_unit(self):
        '''
        Return the value associated with the variable (with units).
        '''
        return Quantity(self.get_value(), self.unit.dimensions)

    def get_addressable_value(self, name, group, level=0):
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
            The group providing the context for the indexing.
        level : int, optional
            How much farther to go down in the stack to find the namespace.

        Returns
        -------
        variable : object
            The variable in an indexable form (without units).
        '''
        return self.get_value()

    def get_addressable_value_with_unit(self, name, group, level=0):
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
            The group providing the context for the indexing.
        level : int, optional
            How much farther to go down in the stack to find the namespace.

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
        description = ('<{classname}(unit={unit}, value={value}, '
                       'dtype={dtype}, scalar={scalar}, constant={constant},'
                       ' is_bool={is_bool}, read_only={read_only})>')
        return description.format(classname=self.__class__.__name__,
                                  unit=repr(self.unit),
                                  value='<value of type %s>' % type(self.value),
                                  dtype=repr(self.dtype),
                                  scalar=repr(self.scalar),
                                  constant=repr(self.constant),
                                  is_bool=repr(self.is_bool),
                                  read_only=repr(self.read_only))


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
    def __init__(self, unit, dtype=None, scalar=False, is_bool=False):
        Variable.__init__(self, unit, value=None, dtype=dtype, scalar=scalar,
                          is_bool=is_bool, read_only=True)


class StochasticVariable(AuxiliaryVariable):
    '''
    An object providing information about a stochastic variable. Automatically
    sets the unit to ``second**-.5``.
    '''
    def __init__(self):
        # The unit of a stochastic variable is always the same
        AuxiliaryVariable.__init__(self, second**(-.5), dtype=np.float64,
                                   scalar=False)


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
    obj : object
        The object storing the variable's value (e.g. a `NeuronGroup`).
    attribute : str
        The name of the attribute storing the variable's value. `attribute` has
        to be an attribute of `obj`.
    constant : bool, optional
        Whether the attribute's value is constant during a run. Defaults to
        ``False``.
    read_only : bool, optional
        Whether this is a read-only variable, i.e. a variable that is set
        internally and cannot be changed by the user (this is used for example
        for the variable ``N``, the number of neurons in a group). Defaults
        to ``False``.
    '''
    def __init__(self, unit, obj, attribute, constant=False, read_only=True):
        # allow for the attribute to not exist yet
        value = getattr(obj, attribute, None)
        
        Variable.__init__(self, unit, value, constant=constant,
                          read_only=read_only)
        #: A reference to the object storing the variable's value         
        self.obj = obj
        #: The name of the attribute storing the variable's value
        self.attribute = attribute

    def get_value(self):
        return getattr(self.obj, self.attribute)

    def __repr__(self):
        description = ('{classname}(unit={unit}, obj={obj}, '
                       'attribute={attribute}, constant={constant})')
        return description.format(classname=self.__class__.__name__,
                                  unit=repr(self.unit),
                                  obj=repr(self.obj),
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
        The group to which this variable belongs
    unit : `Unit`, optional
        The unit to be used for the variable, should be `None` when a variable
         is accessed without units (e.g. when stating `G.var_`).
    level : int, optional
        How much farther to go down in the stack to find the namespace.
    '''

    def __init__(self, name, variable, group, unit=None, level=0):
        self.name = name
        self.variable = variable
        self.group = weakref.proxy(group)
        self.unit = unit
        self.level = level

    def calc_indices(self, item):
        '''
        Return flat indices from to index into state variables from arbitrary
        group specific indices. Thin wrapper around `Group.calc_indices` adding
        special handling for scalar variables.

        Parameters
        ----------
        item : slice, array, int
            The indices to translate.

        Returns
        -------
        indices : `numpy.ndarray`
            The flat indices corresponding to the indices given in `item`.
        '''
        variable = self.variable
        if variable.scalar:
            if not ((isinstance(item, slice) and item == slice(None)) or item == 0 or (hasattr(item, '__len__')
                                                                                           and len(item) == 0)):
                raise IndexError('Variable is a scalar variable.')
            indices = np.array([0])
        else:
            # Translate to an index meaningful for the variable
            # (e.g. from a synaptic [i,j,k] index to the synapse number)
            indices = self.group.calc_indices(item)

        return indices

    def __getitem__(self, item):
        variable = self.variable
        if isinstance(item, basestring):
            values = self.group._get_with_code(self.name, variable, item,
                                               level=self.level+1)
        else:
            indices = self.calc_indices(item)
            if isinstance(variable, Subexpression):
                # For subexpressions, we always have to go through codegen
                values = self.group._get_with_code(self.name, variable, 'True',
                                                   level=self.level+1)[indices]
            else:
                # We are not going via code generation so we have to take care
                # of correct indexing (in particular for subgroups) explicitly
                var_index = self.group.variable_indices[self.name]
                if var_index != '_idx':
                    indices = self.group.variables[var_index].get_value()[indices]
                values = variable.get_value()[indices]

        if self.unit is None or have_same_dimensions(self.unit, Unit(1)):
            return values
        else:
            return Quantity(values, self.unit.dimensions)

    def set_code_with_string_index(self, item, value):
        '''
        Set a variable's value, based on a string condition and a string
        expression for the value.

        Parameters
        ----------
        item : str
            The condition specifying which elements of the variable are to be
            set.
        value : str
            The string expression specifying the value for the variable.
        '''
        check_units = self.unit is not None
        self.group._set_with_code_conditional(self.variable, item, value,
                                              check_units=check_units,
                                              level=self.level + 1)

    def set_code_with_array_index(self, item, value):
        '''
        Set a variable's value, based on a numerical index and a string
        expression for the value.

        Parameters
        ----------
        item : `numpy.ndarray`, slice, tuple
            The indices specifying which elements of the variable are to be
            set.
        value : str
            The string expression specifying the value for the variable.
        '''
        indices = self.calc_indices(item)
        check_units = self.unit is not None
        self.group._set_with_code(self.variable, indices, value,
                                  check_units=check_units,
                                  level=self.level + 1)

    def set_array_with_array_index(self, item, value):
        '''
        Set a variable's value, based on a numerical index and concrete values.

        Parameters
        ----------
        item : `numpy.ndarray`, slice, tuple
            The indices specifying which elements of the variable are to be
            set.
        value : `numpy.ndarray`, `Quantity`, or a broadcastable scalar
            The new values for the variable.
        '''
        indices = self.calc_indices(item)
        if not self.unit is None:
            fail_for_dimension_mismatch(value, self.unit)
        # We are not going via code generation so we have to take care
        # of correct indexing (in particular for subgroups) explicitly
        var_index = self.group.variable_indices[self.variable.name]
        if var_index != '_idx':
            indices = self.group.variables[var_index].get_value()[indices]
        self.variable.value[indices] = value

    def __setitem__(self, item, value):
        if self.variable.read_only:
            raise TypeError('Variable %s is read-only.' % self.name)

        if isinstance(item, slice) and item == slice(None):
            item = 'True'

        # Both index and values are strings, use a single code object do deal
        # with this situation
        if isinstance(value, basestring) and isinstance(item, basestring):
            self.set_code_with_string_index(item, value)
        elif isinstance(item, basestring):
            try:
                float(value)  # only checks for the exception
            except (TypeError, ValueError):
                if not item == 'True':
                    raise TypeError('When setting a variable based on a string '
                                    'index, the value has to be a string or a '
                                    'scalar.')
                else:
                    # Fall back to the general array-array pattern
                    self.set_array_with_array_index(slice(None), value)
                    return
            self.set_code_with_string_index(item, repr(value))
        elif isinstance(value, basestring):
            self.set_code_with_array_index(item, value)
        else:  # No string expressions involved
            self.set_array_with_array_index(item, value)

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
    name : str
        The name of the variable.
    unit : `Unit`
        The unit of the variable
    value : `numpy.ndarray`
        A reference to the array storing the data for the variable.
    group_name : str, optional
        The name of the group to which this variable belongs.
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
    def __init__(self, name, unit, value, group_name=None, constant=False,
                 scalar=False, is_bool=False, read_only=False):

        #: The name of the variable.
        self.name = name

        Variable.__init__(self, unit, value, scalar=scalar,
                          constant=constant, is_bool=is_bool,
                          read_only=read_only)
        #: The reference to the array storing the data for the variable.
        self.value = value

        group_name = '_'+group_name+'_' if group_name is not None else '_'
        #: The name for the array used in generated code
        self.arrayname = '_array' + group_name + name

    def get_value(self):
        return self.value

    def __getitem__(self, item):
        return self.get_value()[item]

    def set_value(self, value, index=None):
        if index is None:
            index = slice(None)
        self.value[index] = value

    def __setitem__(self, item, value):
        self.set_value(value, item)

    def get_addressable_value(self, name, group, level=0):
        return VariableView(name=name, variable=self, group=group, unit=None,
                            level=level+1)

    def get_addressable_value_with_unit(self, name, group, level=0):
        return VariableView(name=name, variable=self, group=group,
                            unit=self.unit, level=level+1)


class DynamicArrayVariable(ArrayVariable):
    '''
    An object providing information about a model variable stored in a dynamic
    array (used in `Synapses`).

    Parameters
    ----------
    name : str
        The name of the variable.
    unit : `Unit`
        The unit of the variable
    value : `numpy.ndarray`
        A reference to the array storing the data for the variable.
    group_name : str, optional
        The name of the group to which this variable belongs.
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

    def __init__(self, name, unit, value, dimensions, group_name=None,
                 constant=False, constant_size=True,
                 scalar=False, is_bool=False, read_only=False):
        #: The number of dimensions
        self.dimensions = dimensions
        if constant and not constant_size:
            raise ValueError('A variable cannot be constant and change in size')
        #: Whether the size of the variable is constant during a run.
        self.constant_size = constant_size
        super(DynamicArrayVariable, self).__init__(name=name,
                                                   unit=unit,
                                                   value=value,
                                                   group_name=group_name,
                                                   constant=constant,
                                                   scalar=scalar,
                                                   is_bool=is_bool,
                                                   read_only=read_only)

    def get_value(self):
        # The actual numpy array is accesible via DynamicArray1D.data
        return self.value.data

    def get_object(self):
        return self.value

    def resize(self, new_size):
        self.value.resize(new_size)


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
    group : `Group`
        The group to which the expression refers.
    dtype : `numpy.dtype`, optional
        The dtype used for the expression. Defaults to
        `core.default_scalar_dtype`.
    is_bool: bool, optional
        Whether this is a boolean variable (also implies it is dimensionless).
        Defaults to ``False``
    '''
    def __init__(self, name, unit, expr, group, dtype=None, is_bool=False):
        #: The name of the subexpression
        self.name = name
        #: The group to which the expression refers
        self.group = group
        Variable.__init__(self, unit, value=None, dtype=dtype,
                          constant=False, scalar=False, is_bool=is_bool,
                          read_only=True)

        #: The expression defining the subexpression
        self.expr = expr.strip()
        #: The identifiers used in the expression
        self.identifiers = get_identifiers(expr)
        
    def get_value(self):
        raise TypeError('get_value should never be called for a Subexpression')

    def get_addressable_value(self, name, group, level=0):
        return VariableView(name=name, variable=self, group=group, unit=None,
                            level=level)

    def get_addressable_value_with_unit(self, name, group, level=0):
        return VariableView(name=name, variable=self, group=group,
                            unit=self.unit, level=level)

    def __contains__(self, var):
        return var in self.identifiers

    def __repr__(self):
        description = ('<{classname}(name={name}, unit={unit}, dtype={dtype}, '
                       'expr={expr}, group=<{group}>, is_bool={is_bool})>')
        return description.format(classname=self.__class__.__name__,
                                  name=repr(self.name),
                                  unit=repr(self.unit),
                                  dtype=repr(self.dtype),
                                  expr=repr(self.expr),
                                  group=repr(self.group.name),
                                  is_bool=repr(self.is_bool))
