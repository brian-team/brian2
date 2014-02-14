import ast
from brian2.utils.stringtools import get_identifiers, deindent
from brian2.parsing.rendering import NodeRenderer


__all__ = ['is_scalar_expression', 'LIONodeRenderer', 'apply_loop_invariant_optimisations']


def is_scalar_expression(expr, scalars, stateless_functions):
    ignore = scalars.union(stateless_functions)
    for id in get_identifiers(str(expr)):
        if id not in ignore:
            return False
    return True


class LIONodeRenderer(NodeRenderer):
    def __init__(self, scalars, stateless_functions):
        self.scalars = scalars
        self.stateless_functions = stateless_functions
        self.optimisations = {}
        self.n = 0
        
    def reset(self):
        self.optimisations.clear()
        
    def render_node(self, node):
        expr = NodeRenderer().render_node(node)
        if is_scalar_expression(expr, self.scalars, self.stateless_functions):
            if expr in self.optimisations:
                name = self.optimisations[expr]
            else:
                self.n += 1
                name = '_scalar_lio_const_'+str(self.n)
                self.optimisations[expr] = name
            return name
        else:
            return NodeRenderer.render_node(self, node)


def apply_loop_invariant_optimisations(code, scalars, stateless_functions):
    scalars = set(scalars)
    stateless_functions = set(stateless_functions)
    code = deindent(code)
    
    renderer = LIONodeRenderer(scalars, stateless_functions)
    
    lines = ast.parse(code).body
    newlines = []
    for line in lines:
        line = renderer.render_node(line)
        newlines.append(line)
    newlines = ['%s = %s' % (name, expr) for expr, name in renderer.optimisations.iteritems()]+newlines
    return '\n'.join(newlines)


if __name__=='__main__':
    code = '''
    x = x*exp(-dt/tau)+y*exp(-dt/tau)
    y = y*exp(-k)+y*exp(-k)*exp(-dt/tau)
    '''
    
    scalars = set(['dt', 'tau'])
    stateless_functions = set(['exp'])
    
    print apply_loop_invariant_optimisations(code, scalars, stateless_functions)
