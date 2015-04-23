'''
Module providing the `Statement` class.
'''

class Statement(object):
    '''
    A single line mathematical statement.
    
    The structure is ``var op expr``.

    Parameters
    ----------
    var : str
        The left hand side of the statement, the value being written to.
    op : str
        The operation, can be any of the standard Python operators (including
        ``+=`` etc.) or a special operator ``:=`` which means you are defining
        a new symbol (whereas ``=`` means you are setting the value of an
        existing symbol).
    expr : str, `Expression`
        The right hand side of the statement.
    dtype : `dtype`
        The numpy dtype of the value or array `var`.
    constant : bool, optional
        Set this flag to ``True`` if the value will not change (only applies for
        ``op==':='``.
    subexpression : bool, optional
        Set this flag to ``True`` if the variable is a subexpression. In some
        languages (e.g. Python) you can use this to save a memory copy, because
        you don't need to do ``lhs[:] = rhs`` but a redefinition ``lhs = rhs``.
    scalar : bool, optional
        Set this flag to ``True`` if `var` and `expr` are scalar.

    Notes
    -----
    Will compute the following attributes:
    
    ``inplace``
        True or False depending if the operation is in-place or not.
    '''
    def __init__(self, var, op, expr, comment, dtype,
                 constant=False, subexpression=False, scalar=False):
        self.var = var.strip()
        self.op = op.strip()
        self.expr = expr
        self.comment = comment
        self.dtype = dtype
        self.constant = constant
        self.subexpression = subexpression
        self.scalar = scalar
        if constant and self.op!=':=':
            raise ValueError("Should not set constant flag for operation "+self.op)
        if op.endswith('=') and op!='=' and op!=':=':
            self.inplace = True
        else:
            self.inplace = False
        
    def __str__(self):
        s = self.var+' '+self.op+' '+str(self.expr)
        if self.constant:
            s += ' (constant)'
        if self.subexpression:
            s += ' (subexpression)'
        if self.inplace:
            s += ' (in-place)'
        if len(self.comment):
            s += ' # ' + self.comment
        return s
    __repr__ = __str__

