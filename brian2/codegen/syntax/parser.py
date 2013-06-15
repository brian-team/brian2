'''
Parses string representations into a tree structure

TODO: how to handle logical operators and/or/not?
TODO: how to handle e.g. '1/2' (always convert to float? provide an option?)
'''

from brian2.utils.stringtools import get_identifiers
from brian2.codegen.parsing import parse_statement as parse_string_statement

__all__ = ['parse_statement', 'parse_expr',
           'OperatorVar', 'AddVar', 'SubVar', 'MulVar', 'DivVar', 'PowVar',
           'AndVar', 'OrVar', 'XorVar',
           'ModVar',
           'NegVar',
           'LeVar', 'LtVar', 'GeVar', 'GtVar', 'EqVar', 'NeVar',
           'FuncVar',
           'ConstVar',
           ]

class Var(object):
    atom = True
    def __init__(self, name):
        self.name = name
        self.items = []
    # Arithemetical operators
    def __mul__(self, other):
        return MulVar(self, other)
    def __rmul__(self, other):
        return MulVar(other, self)
    def __div__(self, other):
        return DivVar(self, other)
    def __rdiv__(self, other):
        return DivVar(other, self)
    def __add__(self, other):
        return AddVar(self, other)
    def __radd__(self, other):
        return AddVar(other, self)
    def __sub__(self, other):
        return SubVar(self, other)
    def __rsub__(self, other):
        return SubVar(other, self)
    def __pow__(self, other):
        return PowVar(self, other)
    def __rpow__(self, other):
        return PowVar(other, self)
    # Logical
    def __and__(self, other):
        return AndVar(self, other)
    def __rand__(self, other):
        return AndVar(other, self)
    def __or__(self, other):
        return OrVar(self, other)
    def __ror__(self, other):
        return OrVar(other, self)
    def __xor__(self, other):
        return XorVar(self, other)
    def __rxor__(self, other):
        return XorVar(other, self)
    def __mod__(self, other):
        return ModVar(self, other)
    def __rmod__(self, other):
        return ModVar(other, self)
    # Comparison
    def __lt__(self, other):
        return LtVar(self, other)
    def __le__(self, other):
        return LeVar(self, other)
    def __gt__(self, other):
        return GtVar(self, other)
    def __ge__(self, other):
        return GeVar(self, other)
    def __eq__(self, other):
        return EqVar(self, other)
    def __ne__(self, other):
        return NeVar(self, other)
    # Unary operators
    def __neg__(self):
        return NegVar(self)
    # Function call
    def __call__(self, *args):
        return FuncVar(self, *args)
    # Invalid
    def __nonzero__(self):
        raise SyntaxError("Cannot use 'and' and 'or' in expressions")
    def __getitem__(self, key):
        raise SyntaxError("Cannot use array syntax in expressions")
    def __setitem__(self, key, val):
        raise SyntaxError("Cannot use array syntax in expressions")
    def __contains__(self, item):
        raise SyntaxError("Cannot use 'in' in expressions")
    def __invert__(self):
        raise SyntaxError("Cannot use '~' in expressions")
    # Convert to string
    def __str__(self):
        return self.name

# TODO: this function and the if len(args)==2 in OperatorVar allow some
# simplification of expressions such as a*b*c instead of (a*b)*c. This
# doesn't cover all cases, but can be improved later.    
def saferep(var):
    if var.atom:
        return str(var)
    else:
        return '('+str(var)+')'

class ConstVar(Var):
    atom = True
    def __init__(self, val):
        self.val = val
        self.items = []
    def __str__(self):
        if self.val<0:
            return '(%s)'%repr(self.val)
        else:
            return repr(self.val)
        
def parse_for_constants(args):
    newargs = []
    for arg in args:
        if not isinstance(arg, Var):
            arg = ConstVar(arg)
        newargs.append(arg)
    return newargs

class OperatorVar(Var):
    atom = False
    def __init__(self, *args):
        args = parse_for_constants(args)
        self.items = list(args)
        if len(args)==2:
            left, right = args
            if right.atom and left.__class__ is self.__class__:
                self.items = left.items+[right]
    def __str__(self):
        return self.op.join(saferep(item) for item in self.items)

class AddVar(OperatorVar):
    op = '+'

class SubVar(OperatorVar):
    op = '-'

class MulVar(OperatorVar):
    op = '*'

class DivVar(OperatorVar):
    op = '/'

class PowVar(OperatorVar):
    op = '**'

class AndVar(OperatorVar):
    op = '&'

class OrVar(OperatorVar):
    op = '|'

class XorVar(OperatorVar):
    op = '^'

class ModVar(OperatorVar):
    op = '%'
    
class LtVar(OperatorVar):
    op = '<'

class LeVar(OperatorVar):
    op = '<='

class GtVar(OperatorVar):
    op = '>'

class GeVar(OperatorVar):
    op = '>='

class EqVar(OperatorVar):
    op = '=='

class NeVar(OperatorVar):
    op = '!='
    
class NegVar(Var):
    atom = False
    def __init__(self, var):
        self.items = [var]
    def __str__(self):
        return '-'+saferep(self.items[0])

class FuncVar(Var):
    atom = True
    def __init__(self, func, *args):
        args = parse_for_constants(args)
        self.items = [func]+list(args)
    func = property(fget=lambda self: self.items[0])
    args = property(fget=lambda self: self.items[1:])
    def __str__(self):
        argslist = ', '.join(str(arg) for arg in self.args)
        return '%s(%s)'%(str(self.func), argslist)

class Statement(object):
    def __init__(self, var, op, expr):
        self.var = var
        self.op = op
        self.expr = expr
    def __str__(self):
        return '%s %s %s'%(str(self.var), self.op, str(self.expr))

def parse_expr(expr):
    varnames = get_identifiers(expr)
    ns = dict((varname, Var(varname)) for varname in varnames)
    expr = eval(expr, ns)
    if not isinstance(expr, Var):
        expr = ConstVar(expr)
    return expr

def parse_statement(code):
    var, op, expr = parse_string_statement(code)
    return Statement(var, op, parse_expr(expr))

if __name__=='__main__':
    print parse_expr('a+1')
    print parse_expr('1.0+a')
    print parse_expr('1+2+x**2')
    x = parse_expr('a+b+c+abs(d)+(-a)**b+2')
    print x
    x = parse_expr('a+b*c+d(e)+f**g+(x==y)')
    print x
    print parse_statement('a+=b+c')
    