import ast

import sympy

from brian2.core.functions import DEFAULT_FUNCTIONS, DEFAULT_CONSTANTS

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

    def __init__(self, use_vectorisation_idx=True):
        self.use_vectorisation_idx = use_vectorisation_idx

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

    def render_NameConstant(self, node):
        return str(node.value)

    def render_Name(self, node):
        return node.id
    
    def render_Num(self, node):
        return repr(node.n)

    def render_Call(self, node):
        if len(node.keywords):
            raise ValueError("Keyword arguments not supported.")
        elif getattr(node, 'starargs', None) is not None:
            raise ValueError("Variable number of arguments not supported")
        elif getattr(node, 'kwargs', None) is not None:
            raise ValueError("Keyword arguments not supported")
        if len(node.args) == 0 and self.use_vectorisation_idx:
            # argument-less function call such as randn() are transformed into
            # randn(_vectorisation_idx) -- this is important for Python code
            # in particular, because it has to return an array of values.
            return '%s(%s)' % (self.render_func(node.func),
                               '_vectorisation_idx')
        else:
            return '%s(%s)' % (self.render_func(node.func),
                           ', '.join(self.render_node(arg) for arg in node.args))

    def render_element_parentheses(self, node):
        '''
        Render an element with parentheses around it or leave them away for
        numbers, names and function calls.
        '''
        if node.__class__.__name__ in ['Name', 'NameConstant']:
            return self.render_node(node)
        elif node.__class__.__name__ == 'Num' and node.n >= 0:
            return self.render_node(node)
        elif node.__class__.__name__ == 'Call':
            return self.render_node(node)
        else:
            return '(%s)' % self.render_node(node)

    def render_BinOp_parentheses(self, left, right, op):
        # Use a simplified checking whether it is possible to omit parentheses:
        # only omit parentheses for numbers, variable names or function calls.
        # This means we still put needless parentheses because we ignore
        # precedence rules, e.g. we write "3 + (4 * 5)" but at least we do
        # not do "(3) + ((4) + (5))"
        return '%s %s %s' % (self.render_element_parentheses(left),
                             self.expression_ops[op.__class__.__name__],
                             self.render_element_parentheses(right))

    def render_BinOp(self, node):
        return self.render_BinOp_parentheses(node.left, node.right, node.op)

    def render_BoolOp(self, node):
        op = node.op
        left = node.values[0]
        remaining = node.values[1:]
        while len(remaining):
            right = remaining[0]
            remaining = remaining[1:]
            s = self.render_BinOp_parentheses(left, right, op)
        op = self.expression_ops[node.op.__class__.__name__]
        return (' '+op+' ').join('%s' % self.render_element_parentheses(v) for v in node.values)

    def render_Compare(self, node):
        if len(node.comparators)>1:
            raise SyntaxError("Can only handle single comparisons like a<b not a<b<c")
        return self.render_BinOp_parentheses(node.left, node.comparators[0], node.ops[0])
        
    def render_UnaryOp(self, node):
        return '%s %s' % (self.expression_ops[node.op.__class__.__name__],
                          self.render_element_parentheses(node.operand))
                
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
          # We'll handle "not" explicitly below
          # Bool ops
          'And': '*',
          'Or': '+',
          })

    def render_UnaryOp(self, node):
        if node.op.__class__.__name__ == 'Not':
            return 'logical_not(%s)' % self.render_node(node.operand)
        else:
            return NodeRenderer.render_UnaryOp(self, node)
    

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
        if node.id in DEFAULT_FUNCTIONS:
            f = DEFAULT_FUNCTIONS[node.id]
            if f.sympy_func is not None and isinstance(f.sympy_func,
                                                       sympy.FunctionClass):
                return '%s' % str(f.sympy_func)
        # special workaround for the "int" function
        if node.id == 'int':
            return 'Function("int_")'
        else:
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
        if node.id in DEFAULT_CONSTANTS:
            c = DEFAULT_CONSTANTS[node.id]
            return '%s' % str(c.sympy_obj)
        elif node.id in ['t', 'dt']:
            return 'Symbol("%s", real=True, positive=True)' % node.id
        else:
            return 'Symbol("%s", real=True)' % node.id

    def render_Num(self, node):
        return 'Float(%s)' % node.n


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
            return '_brian_pow(%s, %s)' % (self.render_node(node.left),
                                    self.render_node(node.right))
        elif node.op.__class__.__name__=='Mod':
            return '_brian_mod(%s, %s)' % (self.render_node(node.left),
                                           self.render_node(node.right))
        else:
            return NodeRenderer.render_BinOp(self, node)

    def render_NameConstant(self, node):
        # In Python 3.4, None, True and False go here
        return {True: 'true',
                False: 'false'}.get(node.value, node.value)

    def render_Name(self, node):
        # Replace Python's True and False with their C++ bool equivalents
        return {'True': 'true',
                'False': 'false',
                'inf': 'INFINITY'}.get(node.id, node.id)

    def render_Assign(self, node):
        return NodeRenderer.render_Assign(self, node)+';'

