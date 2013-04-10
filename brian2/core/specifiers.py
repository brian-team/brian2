'''
Classes used to specify the type of a function, variable or common sub-expression

TODO: have a single global dtype rather than specify for each variable?
'''

from brian2.units.allunits import second

from brian2.utils.stringtools import get_identifiers
from brian2.units.fundamentalunits import is_scalar_type

__all__ = ['Specifier',
           'VariableSpecifier',
           'Value',
           'ReadOnlyValue',
           'StochasticVariable',
           'AttributeValue'
           'ArrayVariable',
           'Subexpression',           
           'Index',
           ]

###############################################################################
# Parent classes
###############################################################################
class Specifier(object):
    '''
    An object providing information about parts of a model (e.g. variables).
    `Specifier` objects are used both to store the information within the model
    (and allow for things like unit checking) and are passed on to code
    generation to specify properties like the dtype.
    
    This class is only used as a parent class for more concrete specifiers.
    
    Parameters
    ----------
    name : str
        The name of the specifier (e.g. the name of the model variable)
    '''
      
    def __init__(self, name):
        #: The name of the thing being specified (e.g. the model variable)
        self.name = name


class VariableSpecifier(Specifier):
    '''
    An object providing information about model variables (including implicit
    variables such as ``t`` or ``xi``).
    
    Parameters
    ----------
    name : str
        The name of the variable.
    unit : `Unit`
        The unit of the variable
    scalar : bool, optional
        Whether the variable is a scalar value (``True``) or vector-valued, i.e.
        defined for every neuron (``False``). Defaults to ``True``.
    constant: bool, optional
        Whether the value of this variable can change during a run. Defaults
        to ``False``.
    See Also
    --------
    Value
    '''
    def __init__(self, name, unit, scalar=True, constant=False):
        Specifier.__init__(self, name)
        
        #: The variable's unit.
        self.unit = unit

        #: Whether the value is a scalar
        self.scalar = scalar

        #: Whether the value is constant during a run
        self.constant = constant

class Value(VariableSpecifier):
    '''
    An object providing information about model variables that have an
    associated value in the model.
    
    Some variables, for example stochastic variables, are not stored anywhere
    in the model itself. They would therefore be represented by a specifier
    that is *not* derived from `Value` but from `VariableSpecifier`. 
    
    Parameters
    ----------
    name : str
        The name of the variable.
    unit : `Unit`
        The unit of the variable
    dtype: `numpy.dtype`
        The dtype used for storing the variable.
    scalar : bool, optional
        Whether the variable is a scalar value (``True``) or vector-valued, i.e.
        defined for every neuron (``False``). Defaults to ``True``.
    constant: bool, optional
        Whether the value of this variable can change during a run. Defaults
        to ``False``.        
    '''
    def __init__(self, name, unit, dtype, scalar=True, constant=False):
        VariableSpecifier.__init__(self, name, unit, scalar, constant)
        #: The dtype used for storing the variable.
        self.dtype = dtype
    
    def get_value(self):
        '''
        Return the value associated with the variable.
        '''
        raise NotImplementedError()
    
    def set_value(self):
        '''
        Set the value associated with the variable.
        '''
        raise NotImplementedError()

###############################################################################
# Concrete classes that are used as specifiers in practice.
###############################################################################

class ReadOnlyValue(Value):
    '''
    An object providing information about a model variable that can only be
    read (e.g. the length of a `NeuronGroup`). It is assumed that the value
    does never change, for changing values use `AttributeValue` instead.
    
    Parameters
    ----------
    name : str
        The name of the variable.
    unit : `Unit`
        The unit of the variable
    dtype: `numpy.dtype`
        The dtype used for storing the variable.
    value : reference to a value of type `dtype`
        Reference to the variable's value

    Raises
    ------
    TypeError
        When trying to use the `set_value` method.
    '''
    def __init__(self, name, unit, dtype, value):
        #: Reference to the variable's value
        self.value = value
        
        scalar = is_scalar_type(value)
        
        Value.__init__(self, name, unit, dtype, scalar, constant=True)

    def get_value(self):
        return self.value

    def set_value(self):
        raise TypeError('The value "%s" is read-only' % self.name)


class StochasticVariable(VariableSpecifier):
    '''
    An object providing information about a stochastic variable. Automatically
    sets the unit to ``second**-.5``.
    
    Parameters
    ----------
    name : str
        The name of the stochastic variable.    
    '''
    def __init__(self, name):
        # The units of stochastic variables is fixed
        VariableSpecifier.__init__(self, name, second**(-.5), scalar=False)


class AttributeValue(ReadOnlyValue):
    '''
    An object providing information about a value saved as an attribute of an
    object. Instead of saving a reference to the value itself, we save the
    name of the attribute. This way, we get the correct value if the attribute
    is overwritten with a new value (e.g. in the case of ``clock.t_``)
    
    The object value has to be accessible by doing ``getattr(obj, attribute)``.
    
    Parameters
    ----------
    name : str
        The name of the variable.
    unit : `Unit`
        The unit of the variable
    dtype: `numpy.dtype`
        The dtype used for storing the variable.
    obj : object
        The object storing the variable's value (e.g. a `NeuronGroup`).
    attribute : str
        The name of the attribute storing the variable's value. `attribute` has
        to be an attribute of `obj`.
    constant : bool, optional
        Whether the attribute's value is constant during a run.
        
    Raises
    ------
    AttributeError
        If `obj` does not have an attribute `attribute`.
        
    '''
    def __init__(self, name, unit, dtype, obj, attribute, constant=False):
        if not hasattr(obj, attribute):
            raise AttributeError(('Object %r does not have an attribute %r, '
                                  'providing the value for %r') %
                                 (obj, attribute, name))
        
        scalar = is_scalar_type(getattr(obj, attribute))
        
        Value.__init__(self, name, unit, dtype, scalar, constant)
        #: A reference to the object storing the variable's value         
        self.obj = obj
        #: The name of the attribute storing the variable's value
        self.attribute = attribute


    def get_value(self):
        return getattr(self.obj, self.attribute)


class ArrayVariable(Value):
    '''
    An object providing information about a model variable stored in an array
    (for example, all state variables). The `index` will be used in the
    generated code (at least in languages such as C++, where the code always
    loops over arrays). Stores a reference to the array name used in the
    generated code, constructed as ``'_array_'`` + ``name``.
    
    For example, for::
    
        ``v = ArrayVariable('_array_v', volt, float64, group.arrays['v'], '_index')``
    
    we would eventually produce C++ code that looked like::
    
        double &v = _array_v[_index];
    
    Parameters
    ----------
    name : str
        The name of the variable.
    unit : `Unit`
        The unit of the variable
    dtype : `numpy.dtype`
        The dtype used for storing the variable.
    array : `numpy.array`
        A reference to the array storing the data for the variable.
    index : str
        The index that will be used in the generated code when looping over the
        variable.
    constant : bool, optional
        Whether the variable's value is constant during a run.
    '''
    def __init__(self, name, unit, dtype, array, index, constant=False):
        Value.__init__(self, name, unit, dtype, scalar=False, constant=constant)
        #: The reference to the array storing the data for the variable.
        self.array = array
        #: The name for the array used in generated code
        self.arrayname = '_array_' + self.name
        #: The name of the index that will be used in the generated code.
        self.index = index

    def get_value(self):
        return self.array

    def set_value(self, value):
        self.array[:] = value


class Subexpression(Value):
    '''
    An object providing information about a static equation in a model
    definition, used as a hint in optimising. Can test if a variable is used
    via ``var in spec``. The specifier is also able to return the result of
    the expression (used in a `StateMonitor`, for example).
    
    Parameters
    ----------
    name : str
        The name of the static equation.
    unit : `Unit`
        The unit of the static equation
    dtype : `numpy.dtype`
        The dtype used for the expression.
    expr : str
        The expression defining the static equation.
    specifiers : dict
        The specifiers dictionary, containing specifiers for the
        model variables used in the expression
    namespace : dict
        The namespace dictionary, containing identifiers for all the external
        variables/functions used in the expression
    '''
    def __init__(self, name, unit, dtype, expr, specifiers, namespace):
        Value.__init__(self, name, unit, dtype, scalar=False)
        #: The expression defining the static equation.
        self.expr = expr.strip()
        #: The identifiers used in the expression
        self.identifiers = get_identifiers(expr)        
        #: Specifiers for the identifiers used in the expression
        self.specifiers = specifiers
        #: Namespace for the identifiers used in the expression
        self.namespace = namespace
        for identifier in self.identifiers:
            if not (identifier in self.specifiers or 
                    identifier in self.namespace):
                raise ValueError(('The provided dictionaries do not '
                                  'contain a specifier for the identifier %s, '
                                  'used in the expression "%s"') %
                                 (identifier, self.expr))

    def get_value(self):
        variable_values = {}
        for identifier in self.identifiers:
            if identifier in self.specifiers:
                variable_values[identifier] = self.specifiers[identifier].get_value()
            else:
                variable_values[identifier] = self.namespace[identifier]
        
        return eval(self.expr, variable_values)

    def __contains__(self, var):
        return var in self.identifiers


class Index(Specifier):
    '''
    An object describing an index variable. You can specify ``iterate_all=True``
    or ``False`` to say whether it is varying over the whole of an input vector
    or a subset. Vectorised langauges (i.e. Python) can use this to optimise the
    reading and writing phase (i.e. you can do ``var = arr`` if
    ``iterate_all==True`` but you need to ``var = whole[idx]`` if
    ``iterate_all==False``).
    
    Parameters
    ----------
    name : str
        The name of the index.
    iterate_all : bool, optional
        Whether the index varies over the whole of an input vector (defaults to
        ``True``).
    '''
    def __init__(self, name, iterate_all=True):
        Specifier.__init__(self, name)
        if bool(iterate_all) != iterate_all:
            raise ValueError(('The "all" argument has to be a bool, '
                              'is type %s instead' % type(all)))
        #: Whether the index varies over the whole of an input vector
        self.iterate_all = iterate_all
