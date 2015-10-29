from brian2.core.functions import DEFAULT_FUNCTIONS, DEFAULT_CONSTANTS
from .rendering import NodeRenderer

__all__ = ['SimplifyingRenderer', 'simplified']

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


class SimplifyingRenderer(NodeRenderer):
    def __init__(self, assumptions=None):
        NodeRenderer.__init__(self, use_vectorisation_idx=False)
        if assumptions is None:
            assumptions = []
        self.ns = defaults_ns.copy()
        for assumption in assumptions:
            try:
                exec assumption in self.ns
            except NameError:
                pass

    def render_expr(self, expr, strip=True):
        oldexpr = ''
        while expr!=oldexpr:
            oldexpr = expr
            expr = NodeRenderer.render_expr(self, expr, strip=strip)
        return expr

    def render_node(self, node):
        expr = NodeRenderer().render_node(node)
        val, evaluated = evaluate_expr(expr, self.ns)
        if evaluated:
            return repr(val)
        else:
            return NodeRenderer.render_node(self, node)

    def render_BinOp(self, node):
        left = self.render_node(node.left)
        right = self.render_node(node.right)
        op = self.expression_ops[node.op.__class__.__name__]
        if op=='*':
            if left=='0' or left=='0.0' or right=='0' or right=='0.0':
                return '0'
            if left=='1' or left=='1.0':
                return right
            if right=='1' or right=='1.0':
                return left
        if op=='+':
            if left=='0' or left=='0.0':
                return right
            if right=='0' or right=='0.0':
                return left
        if op=='-':
            if right=='0' or right=='0.0':
                return left
        if op=='/':
            if right=='1' or right=='1.0':
                return left
            if left=='0' or left=='0.0':
                return '0'
        if op=='**':
            if right=='0' or right=='0.0':
                return '1'
        return NodeRenderer.render_BinOp(self, node)

    def render_Name(self, node):
        if node.id in self.ns and isinstance(self.ns[node.id], (bool, int, float)):
            return repr(self.ns[node.id])
        else:
            return NodeRenderer.render_Name(self, node)


def simplified(expr, assumptions=None):
    return SimplifyingRenderer(assumptions).render_expr(expr)
