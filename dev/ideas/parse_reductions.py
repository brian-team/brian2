from brian2.parsing.rendering import *

__all__ = ['analyse_reductions']


class NestedReductionError(ValueError):
    '''
    Error raised if you try to do a nested reduction (like sum(sum(x))).
    '''
    pass


class AssertNoReductionRenderer(NodeRenderer):
    '''
    Simple AST parser that raises an error if any of the reduction functions are used.
    '''
    def __init__(self, reduction_functions):
        self.reduction_functions = reduction_functions
        
    def render_func(self, node):
        if node.id in self.reduction_functions:
            raise NestedReductionError("You cannot nest calls to reduction functions.")
        return NodeRenderer.render_func(self, node)


class ReductionRenderer(NodeRenderer):
    '''
    AST renderer that keeps track of reduction functions.
    '''
    def __init__(self, reduction_functions):
        self.reduction_functions = set(reduction_functions)
        self.assert_no_reduction_renderer = AssertNoReductionRenderer(self.reduction_functions)
        #: Dictionary storing the output reduction expressions (see notes).
        self.reductions = {}
        
    def render_Call(self, node):
        if node.func.id in self.reduction_functions:
            if len(node.args)!=1 or node.starargs is not None or node.kwargs is not None or len(node.keywords):
                raise ValueError("Reduction functions must take precisely one argument")
            arg = node.args[0]
            expr = self.assert_no_reduction_renderer.render_node(arg)
            name = '_reduction_'+str(len(self.reductions))
            self.reductions[name] = (node.func.id, expr)
            return name
        return NodeRenderer.render_node(self, node)


def analyse_reductions(expr, reduction_functions=['sum', 'max', 'min', 'mean', 'std', 'var']):
    '''
    Function that analyses expressions containing reduction functions
    
    Parameters
    ----------
    expr : str
        The string expression to analyse
    reduction_functions : list of str
        The list of names of the reduction functions.
        
    Returns
    -------
    reductions : dict
        Dictionary whose keys are the names of the reductions, which will be
        things like that ``_reduction_0``, etc. The values are pairs
        ``(func, expr)`` where ``func`` is the name of reduction function,
        and ``expr`` is what the reduction function is applied to. For example,
        ``sum(x**2)`` would give ``('sum', 'x**2')``.
    final_expr : str
        The final expression to be evaluated after the reductions have been
        evaluated. Contains names such as ``_reduction_0`` which correspond
        to the keys in the ``reductions`` dict.
        
    Raises
    ------
    NestedReductionError
        If you attempt to do something like ``sum(sum(x))``.
    ValueError
        If you attempt to do something like ``sum(x, y)``.
        
    Notes
    -----
    
    The following string shows an interpretation (that would be correct code
    for something like numpy)::

        reductions, final_expr = analyse_reductions(expr)
        for k, (id, subexpr) in sorted(reductions.items()):
            print '%s = %s(%s)' % (k, id, subexpr)
        print 'final_value =', final_expr    
    '''
    rr = ReductionRenderer(reduction_functions)
    final_expr = rr.render_expr(expr)
    return rr.reductions, final_expr


if __name__=='__main__':
    import jinja2
    expr = '1+sum(x**2)/max(y**2)'
    reductions, final_expr = analyse_reductions(expr)
    print 'Numpy code:'
    print
    for k, (id, subexpr) in sorted(reductions.items()):
        print '%s = %s(%s)' % (k, id, subexpr)
    print 'final_value =', final_expr
    print
    print 'C++ code'
    print
    # This would be the initialisation step in the language-specific
    # part of the reduction specification
    for k, (id, subexpr) in sorted(reductions.items()):
        if id=='sum':
            print 'double %s = 0;' % k
        elif id=='max':
            print 'double %s = -INFINITY;' % k
    # This would be handled by the template
    print 'for(i=0; i<n; i++) {'
    # This would be handled by the snippet generator called on each subexpr
    print '    // load values etc.'
    # This loop would be in the template
    for k, (id, subexpr) in sorted(reductions.items()):
        # This would have been handled by the snippet generator
        subexpr = CPPNodeRenderer().render_expr(subexpr)
        # This bit again would be handled by the reduction step part of the
        # reduction specification
        # Optionally here there could be an additional computation step for
        # things like var(x) which needs to compute x**2 or alternatively,
        # this transformation could be handled by the ReductionRenderer. The
        # only potential difficulty with that is that it might need you to know
        # the value of N for things like mean.
        if id=='sum':
            print '     %s += %s;' % (k, subexpr)
        elif id=='max':
            t = '_cur'+k
            print '    const double %s = %s;' % (t, subexpr)
            print '    if({t}>{k}) {k} = {t};'.format(k=k, t=t)
    print '}'
    # Optionally here there could be an additional computation step for
    # each reduction (e.g. for mean, thi swould be _reduction_0 = _reduction_0/N
    print 'double final_value = ', CPPNodeRenderer().render_expr(final_expr)+';'
    