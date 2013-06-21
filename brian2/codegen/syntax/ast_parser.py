import ast

class NodeRenderer(object):
    expression_ops = {
      # BinOp
      'Add': '+',
      'Sub': '-',
      'Mult': '*',
      'Div': '/',
      'Pow': '**',
      # Compare
      'Lt': '<',
      'LtE': '<=',
      'Gt': '>',
      'GtE': '>=',
      'Eq': '==',
      'NotEq': '!=',
      # Unary ops
      'Not': 'not',
      'Invert': '~',
      'UAdd': '+',
      'USub': '-',
      # Bool ops
      'And': 'and',
      'Or': 'or',
      }
    
    def render_expr(self, expr):
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

    def render_Name(self, node):
        return node.id
    
    def render_Num(self, node):
        return repr(node.n)

    def render_Call(self, node):
        if len(node.keywords):
            raise ValueError("Keyword arguments not supported.")
        elif node.starargs is not None:
            raise ValueError("*args not supported")
        elif node.kwargs is not None:
            raise ValueError("**kwds not supported")
        return '%s(%s)' % (self.render_node(node.func),
                           ', '.join(self.render_node(arg) for arg in node.args))

    def render_BinOp(self, node):
        return '(%s) %s (%s)'%(self.render_node(node.left),
                               self.expression_ops[node.op.__class__.__name__],
                               self.render_node(node.right))

    def render_BoolOp(self, node):
        op = self.expression_ops[node.op.__class__.__name__]
        return (' '+op+' ').join('(%s)' % self.render_node(v) for v in node.values)

    def render_Compare(self, node):
        if len(node.comparators)>1:
            raise SyntaxError("Can only handle single comparisons like a<b not a<b<c")
        left = node.left
        op = node.ops[0]
        right = node.comparators[0]
        return '(%s) %s (%s)' % (self.render_node(left),
                                 self.expression_ops[op.__class__.__name__],
                                 self.render_node(right))    
        
    def render_UnaryOp(self, node):
        return '%s(%s)' % (self.expression_ops[node.op.__class__.__name__],
                           self.render_node(node.operand))
                
    def render_Assign(self, node):
        if len(node.targets)>1:
            raise SyntaxError("Only support syntax like a=b not a=b=c")
        return '%s = %s' % (self.render_node(node.targets[0]),
                            self.render_node(node.value))


class NumpyNodeRenderer(NodeRenderer):           
    expression_ops = NodeRenderer.expression_ops.copy()
    expression_ops.update({
          # Unary ops
          'Not': 'logical_not',
          'Invert': 'logical_not',
          # Bool ops
          'And': '*',
          'Or': '+',
          })
    

class CPPNodeRenderer(NodeRenderer):
    expression_ops = NodeRenderer.expression_ops.copy()
    expression_ops.update({
          # Unary ops
          'Not': '!',
          'Invert': '!',
          # Bool ops
          'And': '&&',
          'Or': '||',
          })
    
    def render_BinOp(self, node):
        if node.op.__class__.__name__=='Pow':
            return 'pow(%s, %s)' % (self.render_node(node.left),
                                    self.render_node(node.right))
        else:
            return NodeRenderer.render_BinOp(self, node)
        
    def render_Assign(self, node):
        return NodeRenderer.render_Assign(self, node)+';'
    

if __name__=='__main__':
    for renderer in [NodeRenderer(), NumpyNodeRenderer(), CPPNodeRenderer()]:
        name = renderer.__class__.__name__
        print name+'\n'+'='*len(name)
        print renderer.render_expr('a+b*c(d, e)+e**f')
        print renderer.render_expr('a and -b and c and 1.2')
        print renderer.render_code('a=b\nc=d+e')
