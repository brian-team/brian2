'''
Classes used to specify the type of a function, variable or common sub-expression

TODO: have a single global dtype rather than specify for each variable?
'''
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
           ]


def get_dtype(obj):
    if hasattr(obj, 'dtype'):
        return obj.dtype
    else:
        return np.obj2sctype(obj)


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
    '''
    def __init__(self, unit, value=None, dtype=None, scalar=None,
                 constant=False, is_bool=None):
        
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

    def get_value(self):
        '''
        Return the value associated with the variable (without units).
        '''
        if self.value is None:
            raise TypeError('Variable does not have a value')
        else:
            return self.value

    def set_value(self):
        '''
        Set the value associated with the variable.
        '''
        raise NotImplementedError()

    def get_value_with_unit(self):
        '''
        Return the value associated with the variable (with units).
        '''
        return Quantity(self.get_value(), self.unit.dimensions)

    def get_addressable_value(self, level=0):
        '''
        Get the value associated with the variable (without units) that allows
        for indexing
        '''
        return self.get_value()

    def get_addressable_value_with_unit(self, level=0):
        '''
        Get the value associated with the variable (with units) that allows
        for indexing
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

    def __repr__(self):
        description = ('<{classname}(unit={unit}, value={value}, '
                       'dtype={dtype}, scalar={scalar}, constant={constant})>')
        return description.format(classname=self.__class__.__name__,
                                  unit=repr(self.unit),
                                  value='<value of type %s>' % type(self.value),
                                  dtype=repr(self.dtype),
                                  scalar=repr(self.scalar),
                                  constant=repr(self.constant))


class StochasticVariable(Variable):
    '''
    An object providing information about a stochastic variable. Automatically
    sets the unit to ``second**-.5``.

    '''
    def __init__(self):
        # The units of stochastic variables is fixed
        Variable.__init__(self, second**(-.5), dtype=np.float64,
                          scalar=False, constant=False, is_bool=False)


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
    Raises
    ------
    AttributeError
        If `obj` does not have an attribute `attribute`.
        
    '''
    def __init__(self, unit, obj, attribute, constant=False):
        if not hasattr(obj, attribute):
            raise AttributeError('Object %r does not have an attribute %r' %
                                 (obj, attribute))

        value = getattr(obj, attribute)
        
        Variable.__init__(self, unit, value, constant=constant)
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
    template : str
        The template to use when setting variables with a string expression.
    unit : `Unit`, optional
        The unit to be used for the variable, should be `None` when a variable
         is accessed without units (e.g. when stating `G.var_`).
    level : int, optional
        How much farther to go down in the stack to find the namespace.
    '''

    def __init__(self, name, variable, group, template,
                 unit=None, level=0):
        self.name = name
        self.variable = variable
        self.group = group
        self.template = template
        self.unit = unit
        self.level = level

    def __getitem__(self, i):
        variable = self.variable
        if variable.scalar:
            if not (i == slice(None) or i == 0 or (hasattr(i, '__len__') and len(i) == 0)):
                raise IndexError('Variable is a scalar variable.')
            indices = 0
        else:
            indices = self.group.indices[self.group.variable_indices[self.name]][i]
        if self.unit is None or have_same_dimensions(self.unit, Unit(1)):
            return variable.get_value()[indices]
        else:
            return Quantity(variable.get_value()[indices], self.unit.dimensions)

    def __setitem__(self, i, value):
        variable = self.variable
        if variable.scalar:
            if not (i == slice(None) or i == 0 or (hasattr(i, '__len__') and len(i) == 0)):
                raise IndexError('Variable is a scalar variable.')
            indices = np.array([0])
        else:
            indices = self.group.indices[self.group.variable_indices[self.name]][i]
        if isinstance(value, basestring):
            check_units = self.unit is not None
            self.group._set_with_code(variable, indices, value,
                                      template=self.template,
                                      check_units=check_units, level=self.level + 1)
        else:
            if not self.unit is None:
                fail_for_dimension_mismatch(value, self.unit)
            variable.value[indices] = value

    def __array__(self, dtype=None):
        if dtype is not None and dtype != self.variable.dtype:
            raise NotImplementedError('Changing dtype not supported')
        return self[:]

    def __add__(self, other):
        return self[:] + other

    def __sub__(self, other):
        return self[:] - other

    def __mul__(self, other):
        return self[:] * other

    def __div__(self, other):
        return self[:] / other

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
    '''
    def __init__(self, name, unit, value, group_name=None, constant=False,
                 scalar=False, is_bool=False):

        self.name = name

        Variable.__init__(self, unit, value, scalar=scalar,
                          constant=constant, is_bool=is_bool)
        #: The reference to the array storing the data for the variable.
        self.value = value

        group_name = '_'+group_name+'_' if group_name is not None else '_'
        #: The name for the array used in generated code
        self.arrayname = '_array' + group_name + name

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value[:] = value

    def get_addressable_value(self, group, level=0):
        template = getattr(group, '_set_with_code_template',
                           'group_variable_set')
        return VariableView(self.name, self, group, template=template,
                            unit=None, level=level)

    def get_addressable_value_with_unit(self, group, level=0):
        template = getattr(group, '_set_with_code_template',
                           'group_variable_set')
        return VariableView(self.name, self, group, template=template,
                            unit=self.unit, level=level)


class DynamicArrayVariable(ArrayVariable):
    '''
    An object providing information about a model variable stored in a dynamic
    array (used in `Synapses`).
    '''
    
    def get_value(self):
        # The actual numpy array is accesible via DynamicArray1D.data
        return self.value.data


class Subexpression(Variable):
    '''
    An object providing information about a static equation in a model
    definition, used as a hint in optimising. Can test if a variable is used
    via ``var in spec``. The specifier is also able to return the result of
    the expression.
    
    Parameters
    ----------
    unit : `Unit`
        The unit of the static equation
    dtype : `numpy.dtype`
        The dtype used for the expression.
    expr : str
        The expression defining the static equation.
    variables : dict
        The variables dictionary, containing variables for the
        model variables used in the expression
    namespace : dict
        The namespace dictionary, containing identifiers for all the external
        variables/functions used in the expression
    is_bool: bool, optional
        Whether this is a boolean variable (also implies it is dimensionless).
        Defaults to ``False``
    '''
    def __init__(self, unit, dtype, expr, variables, namespace,
                 is_bool=False):
        Variable.__init__(self, unit, value=None, dtype=dtype,
                          constant=False, scalar=False, is_bool=is_bool)

        #: The expression defining the static equation.
        self.expr = expr.strip()
        #: The identifiers used in the expression
        self.identifiers = get_identifiers(expr)        
        #: Specifiers for the identifiers used in the expression
        self.variables = variables
        
        #: The NeuronGroup's namespace for the identifiers used in the
        #: expression
        self.namespace = namespace
        
        #: An additional namespace provided by the run function (and updated
        #: in `NeuronGroup.pre_run`) that is used if the NeuronGroup does not
        #: have an explicitly defined namespace.
        self.additional_namespace = None
        
    def get_value(self):
        raise AssertionError('get_value should never be called for a Subexpression')

    def __contains__(self, var):
        return var in self.identifiers

    def __repr__(self):
        description = ('<{classname}(unit={unit}, dtype={dtype}, '
                       'expr={expr}, variables=<...>, namespace=<....>)>')
        return description.format(classname=self.__class__.__name__,
                                  unit=repr(self.unit),
                                  dtype=repr(self.dtype),
                                  expr=repr(self.expr))        

