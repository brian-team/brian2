from brian2.parsing.bast import brian_ast, BrianASTRenderer, dtype_hierarchy
from brian2.parsing.rendering import NodeRenderer


def optimise_statements(scalar_statements, vector_statements):
    renderer = NodeRenderer()


class Simplifier(BrianASTRenderer):
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

if __name__=='__main__':
    import brian2, ast
    eqs = '''
    x : 1
    y : 1 (shared)
    a : integer
    b : boolean
    c : integer (shared)
    '''
    expr = 'x/1.0'

    G = brian2.NeuronGroup(2, eqs)
    variables = {}
    variables.update(**brian2.DEFAULT_FUNCTIONS)
    variables.update(**brian2.DEFAULT_CONSTANTS)
    variables.update(**G.variables)

    simplifier = Simplifier(variables)
    node = brian_ast(expr, variables)
    node = simplifier.render_node(node)

    print node.dtype, node.scalar
    print NodeRenderer().render_node(node)
