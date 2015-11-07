'''
Brian AST representation
'''

import ast
import numpy
from __builtin__ import all as logical_all # defensive programming against numpy import

__all__ = ['brian_ast']


dtype_hierarchy = {'boolean':0,
                   'integer':1,
                   'float':2,
                   }
for tc, i in dtype_hierarchy.items():
    dtype_hierarchy[i] = tc


def is_boolean(value):
    return isinstance(value, bool)


def is_integer(value):
    return isinstance(value, (int, numpy.integer))


def is_float(value):
    return isinstance(value, (float, numpy.float))


def brian_dtype_from_value(value):
    if is_float(value):
        return 'float'
    elif is_integer(value):
        return 'integer'
    elif is_boolean(value):
        return 'boolean'
    raise TypeError("Unknown dtype for value "+str(value))


def is_boolean_dtype(dtype):
    return numpy.issubdtype(dtype, numpy.bool)


def is_integer_dtype(dtype):
    return numpy.issubdtype(dtype, numpy.integer)


def is_float_dtype(dtype):
    return numpy.issubdtype(dtype, numpy.float)


def brian_dtype_from_dtype(dtype):
    if is_float_dtype(dtype):
        return 'float'
    elif is_integer_dtype(dtype):
        return 'integer'
    elif is_boolean_dtype(dtype):
        return 'boolean'
    raise TypeError("Unknown dtype: "+str(dtype))


def brian_ast(expr, variables):
    node = ast.parse(expr, mode='eval').body
    renderer = BrianASTRenderer(variables)
    return renderer.render_node(node)


class BrianASTRenderer(object):
    def __init__(self, variables):
        self.variables = variables

    def render_node(self, node):
        nodename = node.__class__.__name__
        methname = 'render_'+nodename
        if not hasattr(self, methname):
            raise SyntaxError("Unknown syntax: "+nodename)
        return getattr(self, methname)(node)

    def render_NameConstant(self, node):
        if node.value!='True' and node.value!='False':
            raise SyntaxError("Unknown NameConstant "+node.value)
        node.dtype = 'boolean'
        node.scalar = True
        return node

    def render_Name(self, node):
        if node.id=='True' or node.id=='False':
            node.dtype = 'boolean'
            node.scalar = True
        elif node.id in self.variables:
            var = self.variables[node.id]
            dtype = var.dtype
            node.dtype = brian_dtype_from_dtype(dtype)
            node.scalar = var.scalar
        else: # TODO: handle other names (pi, e, inf)
            raise SyntaxError("Unknown name "+node.id)
        return node

    def render_Num(self, node):
        node.dtype = brian_dtype_from_value(node.n)
        node.scalar = True
        return node

    def render_Call(self, node):
        if len(node.keywords):
            raise ValueError("Keyword arguments not supported.")
        elif node.starargs is not None:
            raise ValueError("Variable number of arguments not supported")
        elif node.kwargs is not None:
            raise ValueError("Keyword arguments not supported")
        for subnode in node.args:
            self.render_node(subnode)
        # TODO: deeper system like the one for units, for now assume all functions return floats
        node.dtype = 'float'
        # Condition for scalarity of function call: stateless and arguments are scalar
        node.scalar = False
        if node.func.id in self.variables:
            funcvar = self.variables[node.func.id]
            if funcvar.stateless:
                node.scalar = logical_all(subnode.scalar for subnode in node.args)
        # we leave node.func because it is an ast.Name object that doesn't have a dtype
        return node

    def render_BinOp(self, node):
        for subnode in [node.left, node.right]:
            self.render_node(subnode)
        # TODO: we could capture some syntax errors here, e.g. bool+bool
        newdtype = dtype_hierarchy[max(dtype_hierarchy[subnode.dtype] for subnode in [node.left, node.right])]
        node.dtype = newdtype
        node.scalar = node.left.scalar and node.right.scalar
        return node

    def render_BoolOp(self, node):
        node.dtype = 'boolean'
        for subnode in node.values:
            self.render_node(subnode)
            if subnode.dtype!='boolean':
                raise TypeError("Boolean operator acting on non-booleans")
        node.scalar = logical_all(subnode.scalar for subnode in node.values)
        return node

    def render_Compare(self, node):
        node.dtype = 'boolean'
        comparators = [node.left]+node.comparators
        for subnode in comparators:
            self.render_node(subnode)
        node.scalar = logical_all(subnode.scalar for subnode in comparators)
        return node

    def render_UnaryOp(self, node):
        self.render_node(node.operand)
        node.dtype = node.operand.dtype
        if node.dtype=='boolean':
            raise TypeError("Unary operators do not apply to boolean types")
        node.scalar = node.operand.scalar
        return node


if __name__=='__main__':
    from brian2 import *
    eqs = '''
    x : 1
    y : 1 (shared)
    a : integer
    b : boolean
    c : integer (shared)
    '''
    expr = 'rand()<3.0'

    G = NeuronGroup(2, eqs)
    variables = {}
    variables.update(**DEFAULT_FUNCTIONS)
    variables.update(**DEFAULT_CONSTANTS)
    variables.update(**G.variables)
    node = brian_ast(expr, variables)

    print node.dtype, node.scalar
