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

__all__ = ['Specifier',
           'Variable',
           'Value',
           'ReadOnlyValue',
           'StochasticVariable',
           'AttributeValue'
           'ArrayVariable',
           'Subexpression',           
           'Index',
           ]


def get_dtype(obj):
    if hasattr(obj, 'dtype'):
        return obj.dtype
    else:
        return np.dtype(type(obj))


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

    def __repr__(self):
        return '%s(name=%r)' % (self.__class__.__name__, self.name)


class Variable(Specifier):
    '''
    An object providing information about model variables (including implicit
    variables such as ``t`` or ``xi``).
    
    Parameters
    ----------
    name : str
        The name of the variable.
    unit : `Unit`
        The unit of the variable. Note that the variable itself (as referenced
        by value) should never have units attached.
    value: reference to the variable value, optional
        Some variables (e.g. stochastic variables) don't have their value
        stored anywhere, they'd pass ``None`` as a value.
    dtype: `numpy.dtype`, optional
        The dtype used for storing the variable. If a
    scalar : bool, optional
        Whether the variable is a scalar value (``True``) or vector-valued, e.g.
        defined for every neuron (``False``). Defaults to ``True``.
    constant: bool, optional
        Whether the value of this variable can change during a run. Defaults
        to ``False``.
    is_bool: bool, optional
        Whether this is a boolean variable (also implies it is dimensionless).
        If no value is given, checks the value itself.
    See Also
    --------
    Value
    '''
    def __init__(self, name, unit, value=None, dtype=None, scalar=None,
                 constant=False, is_bool=None):
        Specifier.__init__(self, name)
        
        #: The variable's unit.
        self.unit = unit

        #: reference to a value of type `dtype`
        self.value = value


        if dtype is None:
            self.dtype = get_dtype(value)
        else:
            value_dtype = get_dtype(value)
            if value is not None and value_dtype != dtype:
                raise TypeError(('Conflicting dtype information for %s, '
                                 'referred value has dtype %r, not '
                                 '%r.') % (name, value_dtype, dtype))
            #: The dtype used for storing the variable.
            self.dtype = dtype

        if is_bool is None:
            if value is None:
                raise TypeError('is_bool needs to be specified if no value is given')
            self.is_bool = value is True or value is False
        else:
            #: Whether this variable is a boolean
            self.is_bool = is_bool

        if is_bool:
            if not have_same_dimensions(unit, 1):
                raise ValueError('Boolean variables can only be dimensionless')

        if scalar is None:
            if value is None:
                raise TypeError('scalar needs to be specified if no value is given')
            self.scalar = is_scalar_type(value)
        else:
            #: Whether the variable is a scalar
            self.scalar = scalar

        #: Whether the variable is constant during a run
        self.constant = constant

    def get_value(self):
        '''
        Return the value associated with the variable.
        '''
        if self.value is None:
            raise TypeError('%s does not have a value' % self.name)
        else:
            return self.value

    def set_value(self):
        '''
        Set the value associated with the variable.
        '''
        raise NotImplementedError()

    def get_value_with_unit(self):
        return Quantity(self.get_value(), self.unit.dimensions)

    def get_addressable_value(self, level=0):
        return self.get_value()

    def get_addressable_value_with_unit(self, level=0):
        return self.get_value_with_unit()

    def get_len(self):
        if self.scalar:
            return 0
        else:
            return len(self.get_value())

    def __repr__(self):
        description = ('<{classname}(name={name}, unit={unit}, value={value}, '
                       'dtype={dtype}, scalar={scalar}, constant={constant})>')
        return description.format(classname=self.__class__.__name__,
                                  name=repr(self.name),
                                  unit=repr(self.unit),
                                  value='<value of type %s>' % type(self.value),
                                  dtype=repr(self.dtype),
                                  scalar=repr(self.scalar),
                                  constant=repr(self.constant))

###############################################################################
# Concrete classes that are used as specifiers in practice.
###############################################################################


class StochasticVariable(Variable):
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
        Variable.__init__(self, name, second**(-.5), dtype=np.float64,
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
    name : str
        The name of the variable.
    unit : `Unit`
        The unit of the variable
    obj : object
        The object storing the variable's value (e.g. a `NeuronGroup`).
    attribute : str
        The name of the attribute storing the variable's value. `attribute` has
        to be an attribute of `obj`.
    constant : bool, optional
        Whether the attribute's value is constant during a run. Defaults to
        ``True``.
    Raises
    ------
    AttributeError
        If `obj` does not have an attribute `attribute`.
        
    '''
    def __init__(self, name, unit, obj, attribute, constant=True):
        if not hasattr(obj, attribute):
            raise AttributeError(('Object %r does not have an attribute %r, '
                                  'providing the value for %r') %
                                 (obj, attribute, name))

        value = getattr(obj, attribute)
        
        Variable.__init__(self, name, unit, value, constant=constant)
        #: A reference to the object storing the variable's value         
        self.obj = obj
        #: The name of the attribute storing the variable's value
        self.attribute = attribute

    def get_value(self):
        return getattr(self.obj, self.attribute)

    def __repr__(self):
        description = ('{classname}(name={name}, unit={unit}, '
                       'obj={obj}, attribute={attribute}, constant={constant})')
        return description.format(classname=self.__class__.__name__,
                                  name=repr(self.name),
                                  unit=repr(self.unit),
                                  obj=repr(self.obj),
                                  attribute=repr(self.attribute),
                                  constant=repr(self.constant))


class VariableView(object):

    def __init__(self, specifier, group, unit=None, level=0):
        self.specifier = specifier
        self.group = group
        self.unit = unit
        self.level = level

#    data = property(lambda self: self.specifier.get_value())

    def __getitem__(self, i):
        spec = self.specifier
        if spec.scalar:
            if not (i == slice(None) or i == 0 or (hasattr(i, '__len__') and len(i) == 0)):
                print 'index', repr(i)
                raise IndexError('Variable %s is a scalar variable.' % spec.name)
            indices = 0
        else:
            indices = self.group.indices[i]
        if self.unit is None or have_same_dimensions(self.unit, Unit(1)):
            return spec.get_value()[indices]
        else:
            return Quantity(spec.get_value()[indices], self.unit.dimensions)

    def __setitem__(self, i, value):
        spec = self.specifier
        if spec.scalar:
            if not (i == slice(None) or i == 0 or (hasattr(i, '__len__') and len(i) == 0)):
                raise IndexError('Variable %s is a scalar variable.' % spec.name)
            indices = np.array([0])
        else:
            indices = self.group.indices[i]
        if isinstance(value, basestring):
            check_units = self.unit is not None
            self.group._set_with_code(spec, indices, value,
                                      check_units, level=self.level + 1)
        else:
            if not self.unit is None:
                fail_for_dimension_mismatch(value, self.unit)
            self.specifier.value[indices] = value

    def __array__(self, dtype=None):
        if dtype is not None and dtype != self.specifier.dtype:
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
            rhs = self.specifier.name + ' + ' + other
        else:
            rhs = self[:] + other
        self[:] = rhs
        return self

    def __isub__(self, other):
        if isinstance(other, basestring):
            rhs = self.specifier.name + ' - ' + other
        else:
            rhs = self[:] - other
        self[:] = rhs
        return self

    def __imul__(self, other):
        if isinstance(other, basestring):
            rhs = self.specifier.name + ' * (' + other + ')'
        else:
            rhs = self[:] * other
        self[:] = rhs
        return self

    def __idiv__(self, other):
        if isinstance(other, basestring):
            rhs = self.specifier.name + ' / (' + other + ')'
        else:
            rhs = self[:] / other
        self[:] = rhs
        return self

    def __repr__(self):
        if self.unit is None or have_same_dimensions(self.unit, Unit(1)):
            return '<%s.%s_: %r>' % (self.group.name, self.specifier.name,
                                     self.specifier.get_value())
        else:
            return '<%s.%s: %r>' % (self.group.name, self.specifier.name,
                                    Quantity(self.specifier.get_value(),
                                             self.unit.dimensions))


class ArrayVariable(Variable):
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
    value : `numpy.array`
        A reference to the array storing the data for the variable.
    index : str
        The index that will be used in the generated code when looping over the
        variable.
    group : `Group`, optional
        A reference to the `Group` that stores this variable, necessary for
        correct indexing.
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
    def __init__(self, name, unit, value, index, group=None,
                 constant=False, scalar=False, is_bool=False):

        self.group = group

        Variable.__init__(self, name, unit, value, scalar=scalar,
                          constant=constant, is_bool=is_bool)
        #: The reference to the array storing the data for the variable.
        self.value = value
        #: The name for the array used in generated code
        groupname = '_'+group.name+'_' if group is not None else '_'
        self.arrayname = '_array' + groupname + self.name
        #: The name of the index that will be used in the generated code.
        self.index = index

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value[:] = value

    def get_addressable_value(self, level=0):
        return VariableView(self, self.group, None, level)

    def get_addressable_value_with_unit(self, level=0):
        return VariableView(self, self.group, self.unit, level)


class DynamicArrayVariable(ArrayVariable):
    '''
    An object providing information about a model variable stored in a dynamic
    array (used in synapses).
    '''
    
    def get_value(self):
        # The actual numpy array is accesible via DynamicArray1D.data
        return self.value.data


class SynapticArrayVariable(DynamicArrayVariable):

    def __init__(self, name, unit, array, index, synapses,
                 constant=False, is_bool=False):
        ArrayVariable.__init__(self, name, unit, array, index, synapses,
                               constant=constant, is_bool=is_bool)
        # Register the object with the `SynapticIndex` object so it gets
        # automatically resized
        synapses.indices.register_variable(self.value)

    def get_addressable_value(self, level=0):
        return VariableView(self, self.group, None, level)

    def get_addressable_value_with_unit(self, level=0):
        return VariableView(self, self.group, self.unit, level)


class Subexpression(Variable):
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
    is_bool: bool, optional
        Whether this is a boolean variable (also implies it is dimensionless).
        Defaults to ``False``
    '''
    def __init__(self, name, unit, dtype, expr, specifiers, namespace,
                 is_bool=False):
        Variable.__init__(self, name, unit, value=None, dtype=dtype,
                          constant=False, scalar=False, is_bool=is_bool)

        #: The expression defining the static equation.
        self.expr = expr.strip()
        #: The identifiers used in the expression
        self.identifiers = get_identifiers(expr)        
        #: Specifiers for the identifiers used in the expression
        self.specifiers = specifiers
        
        #: The NeuronGroup's namespace for the identifiers used in the
        #: expression
        self.namespace = namespace
        
        #: An additional namespace provided by the run function (and updated
        #: in `NeuronGroup.pre_run`) that is used if the NeuronGroup does not
        #: have an explicitly defined namespace.
        self.additional_namespace = None
        
    def get_value(self):
        variable_values = {}
        for identifier in self.identifiers:
            if identifier in self.specifiers:
                variable_values[identifier] = self.specifiers[identifier].get_value()
            else:
                variable_values[identifier] = self.namespace.resolve(identifier,
                                                                     self.additional_namespace,
                                                                     strip_units=True)
        return eval(self.expr, variable_values)

    def __contains__(self, var):
        return var in self.identifiers

    def __repr__(self):
        description = ('<{classname}(name={name}, unit={unit}, dtype={dtype}, '
                       'expr={expr}, specifiers=<...>, namespace=<....>)>')
        return description.format(classname=self.__class__.__name__,
                                  name=repr(self.name),
                                  unit=repr(self.unit),
                                  dtype=repr(self.dtype),
                                  expr=repr(self.expr))        


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

    def __repr__(self):
        return '%s(name=%r, iterate_all=%r)' % (self.__class__.__name__,
                                                self.name,
                                                self.iterate_all)
