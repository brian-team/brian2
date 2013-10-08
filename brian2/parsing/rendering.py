import ast

import sympy

from brian2.core.functions import DEFAULT_FUNCTIONS

__all__ = ['NodeRenderer',
           'NumpyNodeRenderer',
           'CPPNodeRenderer',
           'SympyNodeRenderer'
           ]


class NodeRenderer(object):
    expression_ops = {
      # BinOp
      'Add': '+',
      'Sub': '-',
      'Mult': '*',
      'Div': '/',
      'Pow': '**',
      'Mod': '%',
      # Compare
      'Lt': '<',
      'LtE': '<=',
      'Gt': '>',
      'GtE': '>=',
      'Eq': '==',
      'NotEq': '!=',
      # Unary ops
      'Not': 'not',
      'UAdd': '+',
      'USub': '-',
      # Bool ops
      'And': 'and',
      'Or': 'or',
      # Augmented assign
      'AugAdd': '+=',
      'AugSub': '-=',
      'AugMult': '*=',
      'AugDiv': '/=',
      'AugPow': '**=',
      'AugMod': '%=',
      }
    
    def render_expr(self, expr, strip=True):
        if strip:
            expr = expr.strip()
        node = ast.parse(expr, mode='eval')
        return self.render_node(node.body)

    def render_code(self, code):
        lines = []
        for node in ast.parse(code).body:
            lines.append(self.render_node(node))
        return '\n'.join(lines)

    def render_node(self, node):
        nodename = node.__class__.__name__
        methname = 'render_'+nodename
        if not hasattr(self, methname):
            raise SyntaxError("Unknown syntax: "+nodename)
        return getattr(self, methname)(node)

    def render_func(self, node):
        return self.render_Name(node)

    def render_Name(self, node):
        return node.id
    
    def render_Num(self, node):
        return repr(node.n)

    def render_Call(self, node):
        if len(node.keywords):
            raise ValueError("Keyword arguments not supported.")
        elif node.starargs is not None:
            raise ValueError("Variable number of arguments not supported")
        elif node.kwargs is not None:
            raise ValueError("Keyword arguments not supported")
        if len(node.args) == 0:
            # argument-less function call such as randn() are transformed into
            # randn(_vectorisation_idx) -- this is important for Python code
            # in particular, because it has to return an array of values.
            return '%s(%s)' % (self.render_func(node.func),
                               '_vectorisation_idx')
        else:
            return '%s(%s)' % (self.render_func(node.func),
                           ', '.join(self.render_node(arg) for arg in node.args))

    def render_BinOp_parentheses(self, left, right, op):
        # This function checks whether or not you can ommit parentheses assuming Python
        # precedence relations, hopefully this is the same in C++ and Java, but we'll need
        # to check it
        exprs = ['%s %s %s', '(%s) %s %s', '%s %s (%s)', '(%s) %s (%s)']
        nr = NodeRenderer()
        L = nr.render_node(left)
        R = nr.render_node(right)
        O = NodeRenderer.expression_ops[op.__class__.__name__]
        refexpr = '(%s) %s (%s)' % (L, O, R)
        refexprdump = ast.dump(ast.parse(refexpr))
        for expr in exprs:
            e = expr % (L, O, R)
            if ast.dump(ast.parse(e))==refexprdump:
                return expr % (self.render_node(left),
                               self.expression_ops[op.__class__.__name__],
                               self.render_node(right),
                               )

    def render_BinOp(self, node):
        return self.render_BinOp_parentheses(node.left, node.right, node.op)

    def render_BoolOp(self, node):
        # TODO: for the moment we always parenthesise boolean ops because precedence
        # might be different in different languages and it's safer - also because it's
        # a bit more complicated to write the parenthesis rule
        op = node.op
        left = node.values[0]
        remaining = node.values[1:]
        while len(remaining):
            right = remaining[0]
            remaining = remaining[1:]
            s = self.render_BinOp_parentheses(left, right, op)
        op = self.expression_ops[node.op.__class__.__name__]
        return (' '+op+' ').join('(%s)' % self.render_node(v) for v in node.values)

    def render_Compare(self, node):
        if len(node.comparators)>1:
            raise SyntaxError("Can only handle single comparisons like a<b not a<b<c")
        return self.render_BinOp_parentheses(node.left, node.comparators[0], node.ops[0])
        
    def render_UnaryOp(self, node):
        return '%s(%s)' % (self.expression_ops[node.op.__class__.__name__],
                           self.render_node(node.operand))
                
    def render_Assign(self, node):
        if len(node.targets)>1:
            raise SyntaxError("Only support syntax like a=b not a=b=c")
        return '%s = %s' % (self.render_node(node.targets[0]),
                            self.render_node(node.value))
        
    def render_AugAssign(self, node):
        target = node.target.id
        rhs = self.render_node(node.value)
        op = self.expression_ops['Aug'+node.op.__class__.__name__]
        return '%s %s %s' % (target, op, rhs)


class NumpyNodeRenderer(NodeRenderer):           
    expression_ops = NodeRenderer.expression_ops.copy()
    expression_ops.update({
          # Unary ops
          'Not': 'logical_not',
          # Bool ops
          'And': '*',
          'Or': '+',
          })
    

class SympyNodeRenderer(NodeRenderer):
    expression_ops = NodeRenderer.expression_ops.copy()
    expression_ops.update({
          # Compare
          'Eq': 'Eq',
          'NotEq': 'Ne',
          # Unary ops
          'Not': '~',
          # Bool ops
          'And': '&',
          'Or': '|',
          })

    def render_func(self, node):
        for name, f in DEFAULT_FUNCTIONS.iteritems():
            if name == node.id:
                if f.sympy_func is not None and isinstance(f.sympy_func,
                                                           sympy.FunctionClass):
                    return '%s' % str(f.sympy_func)
        return 'Function("%s")' % node.id

    def render_Compare(self, node):
        if len(node.comparators)>1:
            raise SyntaxError("Can only handle single comparisons like a<b not a<b<c")
        op = node.ops[0]
        if op.__class__.__name__ in ('Eq', 'NotEq'):
            return '%s(%s, %s)' % (self.expression_ops[op.__class__.__name__],
                                   self.render_node(node.left),
                                   self.render_node(node.comparators[0]))
        else:
            return NodeRenderer.render_Compare(self, node)

    def render_Name(self, node):
        if node.id in ['t', 'dt']:
            return 'Symbol("%s", real=True, positive=True)' % node.id
        else:
            return 'Symbol("%s", real=True)' % node.id

    def render_Num(self, node):
        return 'Float(%f)' % node.n


class CPPNodeRenderer(NodeRenderer):
    expression_ops = NodeRenderer.expression_ops.copy()
    expression_ops.update({
          # Unary ops
          'Not': '!',
          # Bool ops
          'And': '&&',
          'Or': '||',
          })
    
    def render_BinOp(self, node):
        if node.op.__class__.__name__=='Pow':
            return 'pow(%s, %s)' % (self.render_node(node.left),
                                    self.render_node(node.right))
        elif node.op.__class__.__name__ == 'Mod':
            # In C, the modulo operator is only defined on integers
            return 'fmod(%s, %s)' % (self.render_node(node.left),
                                     self.render_node(node.right))
        else:
            return NodeRenderer.render_BinOp(self, node)

    def render_Name(self, node):
        # Replace Python's True and False with their C++ bool equivalents
        return {'True': 'true',
                'False': 'false'}.get(node.id, node.id)

    def render_Assign(self, node):
        return NodeRenderer.render_Assign(self, node)+';'

