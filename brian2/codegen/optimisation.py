import ast

from brian2.core.functions import DEFAULT_FUNCTIONS, DEFAULT_CONSTANTS
from brian2.parsing.bast import brian_ast, BrianASTRenderer, dtype_hierarchy
from brian2.parsing.rendering import NodeRenderer


defaults_ns = dict((k, v.pyfunc) for k, v in DEFAULT_FUNCTIONS.iteritems())
defaults_ns.update(**dict((k, v.value) for k, v in DEFAULT_CONSTANTS.iteritems()))


def evaluate_expr(expr, ns=None):
    if ns is None:
        ns = defaults_ns
    try:
        val = eval(expr, ns)
        return val, True
    except NameError:
        return expr, False


def optimise_statements(scalar_statements, vector_statements):
    renderer = NodeRenderer()


class Simplifier(BrianASTRenderer):
    def __init__(self, variables, assumptions=None):
        BrianASTRenderer.__init__(self, variables)
        if assumptions is None:
            assumptions = []
        self.ns = defaults_ns.copy()
        for assumption in assumptions:
            try:
                exec assumption in self.ns
            except NameError:
                pass

    def render_node(self, node):
        node = super(Simplifier, self).render_node(node)
        # can't evaluate vector expressions, so abandon in this case
        if not node.scalar:
            return node
        # try fully evaluating using assumptions
        expr = NodeRenderer().render_node(node)
        val, evaluated = evaluate_expr(expr, self.ns)
        if evaluated:
            if node.dtype=='boolean':
                val = bool(val)
                if hasattr(ast, 'NameConstant'):
                    newnode = ast.NameConstant(val)
                else:
                    newnode = ast.Name(repr(val))
            elif node.dtype=='integer':
                val = int(val)
            else:
                val = float(val)
            if node.dtype!='boolean':
                newnode = ast.Num(val)
            newnode.dtype = node.dtype
            newnode.scalar = True
            return newnode
        return node

    def render_BinOp(self, node):
        left = node.left
        right = node.right
        op = node.op
        # Handle multiplication by 0 or 1
        if op.__class__.__name__=='Mult':
            if left.__class__.__name__=='Num':
                if left.n==0:
                    # must not change the dtype of the output, e.g. handle 0*float->0.0 and 0.0*int->0.0
                    left.dtype = node.dtype
                    if node.dtype=='integer':
                        left.n = 0
                    else:
                        left.n = 0.0
                    return left
                if left.n==1:
                    # only simplify this if the type wouldn't be cast by the operation
                    if dtype_hierarchy[left.dtype]<=dtype_hierarchy[right.dtype]:
                        return right
            if right.__class__.__name__=='Num':
                if right.n==0:
                    # must not change the dtype of the output, e.g. handle 0*float->0.0 and 0.0*int->0.0
                    right.dtype = right.dtype
                    if node.dtype=='integer':
                        right.n = 0
                    else:
                        right.n = 0.0
                    return right
                if right.n==1:
                    # only simplify this if the type wouldn't be cast by the operation
                    if dtype_hierarchy[right.dtype]<=dtype_hierarchy[left.dtype]:
                        return left
        # Handle division by 1, or 0/x
        if op.__class__.__name__=='Div':
            if left.__class__.__name__=='Num':
                if left.n==0:
                    # must not change the dtype of the output, e.g. handle 0/float->0.0 and 0.0/int->0.0
                    left.dtype = node.dtype
                    if node.dtype=='integer':
                        left.n = 0
                    else:
                        left.n = 0.0
                    return left
            if right.__class__.__name__=='Num':
                if right.n==1:
                    # only simplify this if the type wouldn't be cast by the operation
                    if dtype_hierarchy[right.dtype]<=dtype_hierarchy[left.dtype]:
                        return left
        # Handle addition of 0
        if op.__class__.__name__=='Add':
            if left.__class__.__name__=='Num':
                if left.n==0:
                    # only simplify this if the type wouldn't be cast by the operation
                    if dtype_hierarchy[left.dtype]<=dtype_hierarchy[right.dtype]:
                        return right
            if right.__class__.__name__=='Num':
                if right.n==0:
                    # only simplify this if the type wouldn't be cast by the operation
                    if dtype_hierarchy[right.dtype]<=dtype_hierarchy[left.dtype]:
                        return left
        # Handle subtraction of 0
        if op.__class__.__name__=='Sub':
            if right.__class__.__name__=='Num':
                if right.n==0:
                    # only simplify this if the type wouldn't be cast by the operation
                    if dtype_hierarchy[right.dtype]<=dtype_hierarchy[left.dtype]:
                        return left
        return node
