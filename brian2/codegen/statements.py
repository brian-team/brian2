#from expressions import Expression

class Statement(object):
    '''
    A single line mathematical statement.
    
    The structure is ``var op expr``.
    
    ``var``
        The left hand side of the statement, the value being written to, a
        string.
    ``op``
        The operation, can be any of the standard Python operators (including
        ``+=`` etc.) or a special operator ``:=`` which means you are defining
        a new symbol (whereas ``=`` means you are setting the value of an
        existing symbol).
    ``expr``
        A string or an :class:`Expression` object, giving the right hand side
        of the statement.
    ``dtype``
        The numpy dtype of the value or array ``var``.
    ``constant``
        Set this flag to True if the value will not change (only applies for
        ``op==':='``.
    ``subexpression``
        Set this flag to True if the variable is a subexpression. In some
        languages (i.e. Python) you can use this to save a memory copy, because
        you don't need to do ``lhs[:] = rhs`` but a redefinition ``lhs = rhs``.
        
    Will compute the following attributes:
    
    ``inplace``
        True or False depending if the operation is in-place or not.
    '''
    def __init__(self, var, op, expr, dtype,
                 constant=False, subexpression=False):
        self.var = var.strip()
        self.op = op.strip()
        self.expr = expr
        self.dtype = dtype
        self.constant = constant
        self.subexpression = subexpression
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
        return s
    __repr__ = __str__
            
    def translate_to(self, language):
        return language.translate_statement(self)
