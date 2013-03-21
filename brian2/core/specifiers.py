'''
Classes used to specify the type of a function, variable or common sub-expression

TODO: have a single global dtype rather than specify for each variable?
'''

from brian2.units.allunits import second

from brian2.utils.stringtools import get_identifiers

__all__ = ['Specifier',
           'Value',
           'ArrayVariable',
           'Subexpression',
           'StochasticVariable',
           'UnstoredVariable',
           'Index',
           ]
###############################################################################
# Parent classes
###############################################################################
class Specifier(object):
    def __init__(self, name):
        self.name = name


class VariableSpecifier(Specifier):
    def __init__(self, name, unit):
        Specifier.__init__(self, name)
        self.unit = unit


class Value(VariableSpecifier):
    def __init__(self, name, unit, dtype):
        VariableSpecifier.__init__(self, name, unit)
        self.dtype = dtype
    
    def get_value(self):
        raise NotImplementedError()
    
    def set_value(self):
        raise NotImplementedError()

###############################################################################
# Concrete classes that are actually used as specifiers
###############################################################################

class ReadOnlyValue(Value):
    def __init__(self, name, unit, dtype, value):
        Value.__init__(self, name, unit, dtype)
        self.value = value

    def get_value(self):
        return self.value

    def set_value(self):
        raise TypeError('The value "%s" is read-only' % self.name)


class StochasticVariable(VariableSpecifier):
    def __init__(self, name):
        # The units of stochastic variables is fixed
        VariableSpecifier.__init__(self, name, second**(-.5))


class AttributeValue(ReadOnlyValue):
    '''
    A value saved as an attribute of an object. Instead of saving a reference
    to the value itself, we save the name of the attribute. This way, we get
    the correct value if the attribute is overwritten with a new value (e.g.
    in the case of ``clock.t_``)
    '''
    def __init__(self, name, unit, dtype, obj, attribute):
        Value.__init__(self, name, unit, dtype)
        self.obj = obj
        self.attribute = attribute

    def get_value(self):
        return getattr(self.obj, self.attribute)


class ArrayVariable(Value):
    '''
    Used to specify that the variable comes from an array (named ``array``) with
    given ``dtype`` using index variable ``index``. The creation of these
    index variables should be done in the template.
    
    For example, for::
    
        ``v = ArrayVariable('_array_v', float64)``
    
    we would eventually produce C++ code that looked like::
    
        double &v = _array_v[_index];
    '''
    def __init__(self, name, unit, dtype, array, index):
        Value.__init__(self, name, unit, dtype)
        self.array = array
        self.arrayname = '_array_' + self.name
        self.index = index

    def get_value(self):
        return self.array

    def set_value(self, value):
        self.array[:] = value

class Subexpression(VariableSpecifier):
    '''
    Sub-expression, comes from user-defined equation, used as a hint
    in optimising. Can test if a variable is used via ``var in spec``.
    The list of identifiers is given in the ``identifiers`` attribute, and
    the full expression in the ``expr`` attribute.
    '''
    def __init__(self, name, unit, expr):
        VariableSpecifier.__init__(self, name, unit)
        self.expr = expr.strip()
        self.identifiers = get_identifiers(expr)

    def __contains__(self, var):
        return var in self.identifiers


class Index(Specifier):
    '''
    The variable is an index, you can specify ``all=True`` or ``False`` to say
    whether it is varying over the whole of an input vector or a subset.
    Vectorised langauges (i.e. Python) can use this to optimise the reading
    and writing phase (i.e. you can do ``var = arr`` if ``all==True`` but you
    need to ``var = arr[idx]`` if ``all=False``).
    '''
    def __init__(self, name, all=True):
        Specifier.__init__(self, name)
        self.all = all

