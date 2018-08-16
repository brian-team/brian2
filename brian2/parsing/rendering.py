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
      'FloorDiv': '//',
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
        try:
            return getattr(self, methname)(node)
        except AttributeError:
            raise SyntaxError("Unknown syntax: " + nodename)

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
        op_class = op.__class__.__name__
        # Give a more useful error message when using bit-wise operators
        if op_class in ['BitXor', 'BitAnd', 'BitOr']:
            correction = {'BitXor': ('^', '**'),
                          'BitAnd': ('&', 'and'),
                          'BitOr': ('|', 'or')}.get(op_class)
            raise SyntaxError('The operator "{}" is not supported, use "{}" '
                              'instead.'.format(correction[0], correction[1]))
        return '%s %s %s' % (self.render_element_parentheses(left),
                             self.expression_ops[op_class],
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
          'And': '&',
          'Or': '|',
          })

    def render_UnaryOp(self, node):
        if node.op.__class__.__name__ == 'Not':
            return 'logical_not(%s)' % self.render_node(node.operand)
        else:
            return NodeRenderer.render_UnaryOp(self, node)
    

class SympyNodeRenderer(NodeRenderer):
    expression_ops = {
      'Add': sympy.Add,
      'Mult': sympy.Mul,
      'Pow': sympy.Pow,
      'Mod': sympy.Mod,
      # Compare
      'Lt': sympy.StrictLessThan,
      'LtE': sympy.LessThan,
      'Gt': sympy.StrictGreaterThan,
      'GtE': sympy.GreaterThan,
      'Eq': sympy.Eq,
      'NotEq': sympy.Ne,
      # Unary ops are handled manually
      # Bool ops
      'And': sympy.And,
      'Or': sympy.Or}

    def render_func(self, node):
        if node.id in DEFAULT_FUNCTIONS:
            f = DEFAULT_FUNCTIONS[node.id]
            if f.sympy_func is not None and isinstance(f.sympy_func,
                                                       sympy.FunctionClass):
                return f.sympy_func
        # special workaround for the "int" function
        if node.id == 'int':
            return sympy.Function("int_")
        else:
            return sympy.Function(node.id)

    def render_Call(self, node):
        if len(node.keywords):
            raise ValueError("Keyword arguments not supported.")
        elif getattr(node, 'starargs', None) is not None:
            raise ValueError("Variable number of arguments not supported")
        elif getattr(node, 'kwargs', None) is not None:
            raise ValueError("Keyword arguments not supported")
        elif len(node.args) == 0:
            return self.render_func(node.func)(sympy.Symbol('_vectorisation_idx'))
        else:
            return self.render_func(node.func)(*(self.render_node(arg)
                                                 for arg in node.args))

    def render_Compare(self, node):
        if len(node.comparators)>1:
            raise SyntaxError("Can only handle single comparisons like a<b not a<b<c")
        op = node.ops[0]
        return self.expression_ops[op.__class__.__name__](self.render_node(node.left), self.render_node(node.comparators[0]))

    def render_Name(self, node):
        if node.id in DEFAULT_CONSTANTS:
            c = DEFAULT_CONSTANTS[node.id]
            return c.sympy_obj
        elif node.id in ['t', 'dt']:
            return sympy.Symbol(node.id, real=True, positive=True)
        else:
            return sympy.Symbol(node.id, real=True)

    def render_NameConstant(self, node):
        if node.value in [True, False]:
            return node.value
        else:
            return str(node.value)

    def render_Num(self, node):
        return sympy.Float(node.n)

    def render_BinOp(self, node):
        op_name = node.op.__class__.__name__
        # Sympy implements division and subtraction as multiplication/addition
        if op_name == 'Div':
            op = self.expression_ops['Mult']
            return op(self.render_node(node.left),
                      1 / self.render_node(node.right))
        elif op_name == 'FloorDiv':
            op = self.expression_ops['Mult']
            left = self.render_node(node.left)
            right = self.render_node(node.right)
            return sympy.floor(op(left, 1 / right))
        elif op_name == 'Sub':
            op = self.expression_ops['Add']
            return op(self.render_node(node.left),
                      -self.render_node(node.right))
        else:
            op = self.expression_ops[op_name]
            return op(self.render_node(node.left), self.render_node(node.right))

    def render_BoolOp(self, node):
        op = self.expression_ops[node.op.__class__.__name__]
        return op(*(self.render_node(value) for value in node.values))

    def render_UnaryOp(self, node):
        op_name = node.op.__class__.__name__
        if op_name == 'UAdd':
            # Nothing to do
            return self.render_node(node.operand)
        elif op_name == 'USub':
            return -self.render_node(node.operand)
        elif op_name == 'Not':
            return sympy.Not(self.render_node(node.operand))
        else:
            raise ValueError('Unknown unary operator: ' + op_name)


class CPPNodeRenderer(NodeRenderer):
    expression_ops = NodeRenderer.expression_ops.copy()
    expression_ops.update({
          # Unary ops
          'Not': '!',
          # Bool ops
          'And': '&&',
          'Or': '||',
          # C does not have a floor division operator (but see render_BinOp)
          'FloorDiv': '/',
          })
    
    def render_BinOp(self, node):
        if node.op.__class__.__name__ == 'Pow':
            return '_brian_pow(%s, %s)' % (self.render_node(node.left),
                                    self.render_node(node.right))
        elif node.op.__class__.__name__ == 'Mod':
            return '_brian_mod(%s, %s)' % (self.render_node(node.left),
                                           self.render_node(node.right))
        elif node.op.__class__.__name__ == 'Div':
            # C uses integer division, this is a quick and dirty way to assure
            # it uses floating point division for integers
            return '1.0f*%s/%s' % (self.render_element_parentheses(node.left),
                                   self.render_element_parentheses(node.right))
        elif node.op.__class__.__name__ == 'FloorDiv':
            return '_brian_floordiv(%s, %s)' % (self.render_node(node.left),
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

