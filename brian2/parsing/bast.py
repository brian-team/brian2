'''
Brian AST representation

This is a standard Python AST representation with additional information added.
'''

import ast
import weakref

import numpy
from __builtin__ import all as logical_all # defensive programming against numpy import

from brian2.parsing.rendering import NodeRenderer
from brian2.utils.logger import get_logger

__all__ = ['brian_ast', 'BrianASTRenderer', 'dtype_hierarchy']


logger = get_logger(__name__)

# This codifies the idea that operations involving e.g. boolean and integer will end up
# as integer. In general the output type will be the max of the hierarchy values here.
dtype_hierarchy = {'boolean': 0,
                   'integer': 1,
                   'float': 2,
                   }
# This is just so you can invert from number to string
for tc, i in dtype_hierarchy.items():
    dtype_hierarchy[i] = tc

def is_boolean(value):
    return isinstance(value, bool)


def is_integer(value):
    return isinstance(value, (int, numpy.integer))


def is_float(value):
    return isinstance(value, (float, numpy.float32, numpy.float64))


def brian_dtype_from_value(value):
    '''
    Returns 'boolean', 'integer' or 'float'
    '''
    if is_float(value):
        return 'float'
    elif is_integer(value):
        return 'integer'
    elif is_boolean(value):
        return 'boolean'
    raise TypeError("Unknown dtype for value "+str(value))

# The following functions are called very often during the optimisation process
# so we don't use numpy.issubdtype but instead a precalculated list of all
# standard types

bool_dtype =numpy.dtype(numpy.bool)
def is_boolean_dtype(obj):
    return numpy.dtype(obj) is bool_dtype

integer_dtypes = {numpy.dtype(c) for c in numpy.typecodes['AllInteger']}
def is_integer_dtype(obj):
    return numpy.dtype(obj) in integer_dtypes

float_dtypes = {numpy.dtype(c) for c in numpy.typecodes['AllFloat']}
def is_float_dtype(obj):
    return numpy.dtype(obj) in float_dtypes


def brian_dtype_from_dtype(dtype):
    '''
    Returns 'boolean', 'integer' or 'float'
    '''
    if is_float_dtype(dtype):
        return 'float'
    elif is_integer_dtype(dtype):
        return 'integer'
    elif is_boolean_dtype(dtype):
        return 'boolean'
    raise TypeError("Unknown dtype: "+str(dtype))


def brian_ast(expr, variables):
    '''
    Returns an AST tree representation with additional information

    Each node will be a standard Python ``ast`` node with the
    following additional attributes:

    ``dtype``
        One of ``'boolean'``, ``'integer'`` or ``'float'``, referring to the data type
        of the value of this node.
    ``scalar``
        Either ``True`` or ``False`` if the node uses any vector-valued variables.
    ``complexity``
        An integer representation of the computational complexity of the node. This
        is a very rough representation used for things like ``2*(x+y)`` is less
        complex than ``2*x+2*y`` and ``exp(x)`` is more complex than ``2*x`` but
        shouldn't be relied on for fine distinctions between expressions.

    Parameters
    ----------
    expr : str
        The expression to convert into an AST representation
    variables : dict
        The dictionary of `Variable` objects used in the expression.
    '''
    node = ast.parse(expr, mode='eval').body
    renderer = BrianASTRenderer(variables)
    return renderer.render_node(node)


class BrianASTRenderer(object):
    '''
    This class is modelled after `NodeRenderer` - see there for details.
    '''
    def __init__(self, variables, copy_variables=True):
        if copy_variables:
            self.variables = variables.copy()
        else:
            self.variables = variables

    def render_node(self, node):
        nodename = node.__class__.__name__
        methname = 'render_'+nodename
        try:
            return getattr(self, methname)(node)
        except AttributeError:
            raise SyntaxError("Unknown syntax: " + nodename)

    def render_NameConstant(self, node):
        if node.value is not True and node.value is not False:
            raise SyntaxError("Unknown NameConstant "+str(node.value))
        # NameConstant only used for True and False and None, and we don't support None
        node.dtype = 'boolean'
        node.scalar = True
        node.complexity = 0
        node.stateless = True
        return node

    def render_Name(self, node):
        node.complexity = 0
        if node.id=='True' or node.id=='False':
            node.dtype = 'boolean'
            node.scalar = True
        elif node.id in self.variables:
            var = self.variables[node.id]
            dtype = var.dtype
            node.dtype = brian_dtype_from_dtype(dtype)
            node.scalar = var.scalar
        else: # don't think we need to handle other names (pi, e, inf)?
            node.dtype = 'float'
            node.scalar = True # I think this assumption is OK, but not certain
        node.stateless = True
        return node

    def render_Num(self, node):
        node.complexity = 0
        node.dtype = brian_dtype_from_value(node.n)
        node.scalar = True
        node.stateless = True
        return node

    def render_Call(self, node):
        if len(node.keywords):
            raise ValueError("Keyword arguments not supported.")
        elif getattr(node, 'starargs', None) is not None:
            raise ValueError("Variable number of arguments not supported")
        elif getattr(node, 'kwargs', None) is not None:
            raise ValueError("Keyword arguments not supported")
        args = []
        for subnode in node.args:
            subnode.parent = weakref.proxy(node)
            subnode = self.render_node(subnode)
            args.append(subnode)
        node.args = args
        node.dtype = 'float' # default dtype
        # Condition for scalarity of function call: stateless and arguments are scalar
        node.scalar = False
        if node.func.id in self.variables:
            funcvar = self.variables[node.func.id]
            # sometimes this attribute doesn't exist, if so assume it's not stateless
            node.stateless = getattr(funcvar, 'stateless', False)
            if node.stateless:
                node.scalar = logical_all(subnode.scalar for subnode in node.args)
            # check that argument types are valid
            node_arg_types = [subnode.dtype for subnode in node.args]
            for subnode, argtype in zip(node.args, funcvar._arg_types):
                if argtype!='any' and argtype!=subnode.dtype:
                    raise TypeError("Function %s takes arguments with types %s but "
                                    "received %s" % (node.func.id, funcvar._arg_types, node_arg_types))
            # compute return type
            return_type = funcvar._return_type
            if return_type=='highest':
                return_type = dtype_hierarchy[max(dtype_hierarchy[nat] for nat in node_arg_types)]
            node.dtype = return_type
        else:
            node.stateless = False
        # we leave node.func because it is an ast.Name object that doesn't have a dtype
        # TODO: variable complexity for function calls?
        node.complexity = 20+sum(subnode.complexity for subnode in node.args)
        return node

    def render_BinOp(self, node):
        node.left.parent = weakref.proxy(node)
        node.right.parent = weakref.proxy(node)
        node.left = self.render_node(node.left)
        node.right = self.render_node(node.right)
        # TODO: we could capture some syntax errors here, e.g. bool+bool
        # captures, e.g. int+float->float
        newdtype = dtype_hierarchy[max(dtype_hierarchy[subnode.dtype] for subnode in [node.left, node.right])]
        if node.op.__class__.__name__ == 'Div':
            # Division turns integers into floating point values
            newdtype = 'float'
            # Give a warning if the code uses floating point division where it
            # previously might have used floor division
            if node.left.dtype == node.right.dtype == 'integer':
                # This would have led to floor division in earlier versions of
                # Brian (except for the numpy target on Python 3)
                # Ignore cases where the user already took care of this by
                # wrapping the result of the division in int(...) or
                # floor(...)
                if not (hasattr(node, 'parent') and
                        node.parent.__class__.__name__ == 'Call' and
                        node.parent.func.id in ['int', 'floor']):
                    rendered_expr = NodeRenderer().render_node(node)
                    msg = ('The expression "{}" divides two integer values. '
                           'In previous versions of Brian, this would have '
                           'used either an integer ("flooring") or a floating '
                           'point division, depending on the Python version '
                           'and the code generation target. In the current '
                           'version, it always uses a floating point '
                           'division. Explicitly ask for an  integer division '
                           '("//"), or turn one of the operands into a '
                           'floating point value (e.g. replace "1/2" by '
                           '"1.0/2") to no longer receive this '
                           'warning.'.format(rendered_expr))
                    logger.warn(msg, 'floating_point_division', once=True)
        node.dtype = newdtype
        node.scalar = node.left.scalar and node.right.scalar
        node.complexity = 1+node.left.complexity+node.right.complexity
        node.stateless = node.left.stateless and node.right.stateless
        return node

    def render_BoolOp(self, node):
        values = []
        for subnode in node.values:
            subnode.parent = node
            subnode = self.render_node(subnode)
            values.append(subnode)
        node.values = values
        node.dtype = 'boolean'
        for subnode in node.values:
            if subnode.dtype!='boolean':
                raise TypeError("Boolean operator acting on non-booleans")
        node.scalar = logical_all(subnode.scalar for subnode in node.values)
        node.complexity = 1+sum(subnode.complexity for subnode in node.values)
        node.stateless = logical_all(subnode.stateless
                                     for subnode in node.values)
        return node

    def render_Compare(self, node):
        node.left = self.render_node(node.left)
        comparators = []
        for subnode in node.comparators:
            subnode.parent = node
            subnode = self.render_node(subnode)
            comparators.append(subnode)
        node.comparators = comparators
        node.dtype = 'boolean'
        comparators = [node.left]+node.comparators
        node.scalar = logical_all(subnode.scalar for subnode in comparators)
        node.complexity = 1+sum(subnode.complexity for subnode in comparators)
        node.stateless = node.left.stateless and all(c.stateless
                                                     for c in node.comparators)
        return node

    def render_UnaryOp(self, node):
        node.operand.parent = node
        node.operand = self.render_node(node.operand)
        node.dtype = node.operand.dtype
        if node.dtype=='boolean' and node.op.__class__.__name__ != 'Not':
            raise TypeError("Unary operator %s does not apply to boolean types" % node.op.__class__.__name__)
        node.scalar = node.operand.scalar
        node.complexity = 1+node.operand.complexity
        node.stateless = node.operand.stateless
        return node


if __name__=='__main__':
    import brian2
    eqs = '''
    x : 1
    y : 1 (shared)
    a : integer
    b : boolean
    c : integer (shared)
    '''
    expr = 'x<3.0+1.0'

    G = brian2.NeuronGroup(2, eqs)
    variables = {}
    variables.update(**brian2.DEFAULT_FUNCTIONS)
    variables.update(**brian2.DEFAULT_CONSTANTS)
    variables.update(**G.variables)
    node = brian_ast(expr, variables)

    print node.dtype, node.scalar, node.complexity
