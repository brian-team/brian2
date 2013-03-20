'''
Classes used to specify the type of a function, variable or common sub-expression

TODO: have a single global dtype rather than specify for each variable?
'''

from brian2.units.allunits import second

from brian2.utils.stringtools import get_identifiers

__all__ = ['Specifier',
           'Value',
           'ArrayVariable',
           'OutputVariable',
           'Subexpression',
           'StochasticVariable',
           'UnstoredVariable',
           'Index',
           ]

class Specifier(object):
    pass

class UnstoredVariable(Specifier):
    '''
        Superclass for specifiers that do not have any storage associated with
        them. One example are stochastic variables, that need specifiers (to
        allow for unit checking, for example) but are only temporarily
        created within the state update loop.
    '''
    pass

class Value(Specifier):
    def __init__(self, dtype, value):
        self.dtype = dtype
        self.value = value

    def get_value(self):
        return self.value

    def set_value(self):
        raise TypeError()

class StochasticVariable(UnstoredVariable):
    def __init__(self, name, dtype):
        self.name = name
        # The units of stochastic variables is fixed
        self.unit = second**(-.5)
        self.dtype = dtype

class AttributeValue(Value):
    '''
    A value saved as an attribute of an object. Instead of saving a reference
    to the value itself, we save the name of the attribute. This way, we get
    the correct value if the attribute is overwritten with a new value (e.g.
    in the case of ``clock.t_``)
    '''
    def __init__(self, dtype, obj, attribute, unit):
        self.dtype = dtype
        self.obj = obj
        self.attribute = attribute
        self.unit = unit

    def get_value(self):
        return getattr(self.obj, self.attribute)

    def set_value(self, value):
        setattr(self.obj, self.attribute, value)


class ArrayVariable(Specifier):
    '''
    Used to specify that the variable comes from an array (named ``array``) with
    given ``dtype`` using index variable ``index``. The creation of these
    index variables should be done in the template.
    
    For example, for::
    
        ``v = ArrayVariable('_array_v', float64)``
    
    we would eventually produce C++ code that looked like::
    
        double &v = _array_v[_index];
    '''
    def __init__(self, name, dtype, array, index, unit=None):
        self.name = name
        self.array = array
        self.arrayname = '_array_' + self.name
        self.index = index
        self.dtype = dtype
        self.unit = unit

    def get_value(self):
        return self.array

    def set_value(self, value):
        self.array[:] = value

class OutputVariable(UnstoredVariable):
    '''
    Used to specify that this variable is used as an output of the code, with
    given ``dtype``.
    '''
    def __init__(self, dtype):
        self.dtype = dtype

class Subexpression(Specifier):
    '''
    Sub-expression, comes from user-defined equation, used as a hint
    in optimising. Can test if a variable is used via ``var in spec``.
    The list of identifiers is given in the ``identifiers`` attribute, and
    the full expression in the ``expr`` attribute.
    '''
    def __init__(self, expr, unit):
        self.expr = expr.strip()
        self.identifiers = get_identifiers(expr)
        self.unit = unit

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
    def __init__(self, all=True):
        self.all = all

if __name__ == '__main__':
    spec = Subexpression('x*y+z')
    print 'y' in spec, 'w' in spec
