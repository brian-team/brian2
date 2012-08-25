'''
Classes used to specify the type of a function, variable or common sub-expression

TODO: have a single global dtype rather than specify for each variable?
'''

from brian2.utils.stringtools import get_identifiers

__all__ = ['Specifier',
           'Function',
           'Value',
           'ArrayVariable',
           'OutputVariable',
           'Subexpression',
           'Index',
           ]

class Specifier(object):
    pass

class Function(Specifier):
    pass

class Value(Specifier):
    def __init__(self, dtype):
        self.dtype = dtype
        
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
    def __init__(self, array, index, dtype):
        self.array = array
        self.index = index
        self.dtype = dtype

class OutputVariable(Specifier):
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
    def __init__(self, expr):
        self.expr = expr.strip()
        self.identifiers = set(get_identifiers(expr))
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
    
if __name__=='__main__':
    spec = Subexpression('x*y+z')
    print 'y' in spec, 'w' in spec
