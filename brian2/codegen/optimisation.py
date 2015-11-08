# TODO: handle correctly things like ((a*2)*3)*4 -> a*24
# we can do this by checking as part of Mult handling that all subnodes are mults or divs, ...

import ast
from collections import OrderedDict
import copy
import itertools

from brian2.core.functions import DEFAULT_FUNCTIONS, DEFAULT_CONSTANTS
from brian2.core.variables import Variable
from brian2.parsing.bast import (brian_ast, BrianASTRenderer, dtype_hierarchy, is_boolean_dtype,
                                 brian_dtype_from_dtype, brian_dtype_from_value)
from brian2.parsing.rendering import NodeRenderer
from brian2.utils.stringtools import get_identifiers, word_substitute

from .statements import Statement

defaults_ns = dict((k, v.pyfunc) for k, v in DEFAULT_FUNCTIONS.iteritems())
defaults_ns.update(**dict((k, v.value) for k, v in DEFAULT_CONSTANTS.iteritems()))


__all__ = ['optimise_statements', 'ArithmeticSimplifier', 'Simplifier']


def evaluate_expr(expr, ns=None):
    if ns is None:
        ns = defaults_ns
    try:
        val = eval(expr, ns)
        return val, True
    except NameError:
        return expr, False


def optimise_statements(scalar_statements, vector_statements, variables):
    boolvars = dict((k, v) for k, v in variables.iteritems()
                    if hasattr(v, 'dtype') and brian_dtype_from_dtype(v.dtype)=='boolean')
    simplifier = Simplifier(variables, scalar_statements)
    new_vector_statements = []
    for stmt in vector_statements:
        new_expr = simplifier.render_expr(stmt.expr)
        new_stmt = Statement(stmt.var, stmt.op, new_expr, stmt.comment,
                             dtype=stmt.dtype,
                             constant=stmt.constant,
                             subexpression=stmt.subexpression,
                             scalar=stmt.scalar)
        #complexity_std = expression_complexity(expr_std)
        idents = get_identifiers(new_expr)
        used_boolvars = [var for var in boolvars.iterkeys() if var in idents]
        if len(used_boolvars):
            bool_space = [[False, True] for var in used_boolvars]
            expanded_expressions = {}
            #complexities = {}
            for bool_vals in itertools.product(*bool_space):
                subs = dict((var, str(val)) for var, val in zip(used_boolvars, bool_vals))
                curexpr = word_substitute(new_expr, subs)
                curexpr = simplifier.render_expr(curexpr)
                key = tuple((var, val) for var, val in zip(used_boolvars, bool_vals))
                expanded_expressions[key] = curexpr
                #complexities[key] = expression_complexity(curexpr)
                # print ', '.join('%s=%s'%(k, v) for k, v in key)
                # print '-> ', curexpr
            new_stmt.used_boolean_variables = used_boolvars
            new_stmt.boolean_simplified_expressions = expanded_expressions
        new_vector_statements.append(new_stmt)
    new_scalar_statements = copy.copy(scalar_statements)
    for expr, name in simplifier.loop_invariants.iteritems():
        dtype_name = simplifier.loop_invariant_dtypes[name]
        if dtype_name=='boolean':
            dtype = bool
        elif dtype_name=='integer':
            dtype = int
        else:
            dtype = float
        new_stmt = Statement(name, ':=', expr, '',
                             dtype=dtype,
                             constant=True,
                             subexpression=False,
                             scalar=True)
        new_scalar_statements.append(new_stmt)
    return new_scalar_statements, new_vector_statements


def create_assumptions_namespace(assumptions):
    ns = defaults_ns.copy()
    for assumption in assumptions:
        try:
            exec assumption in ns
        except NameError:
            pass
    return ns


class ArithmeticSimplifier(BrianASTRenderer):
    def __init__(self, variables, assumptions=None):
        BrianASTRenderer.__init__(self, variables)
        if assumptions is None:
            assumptions = []
        self.assumptions = assumptions
        self.assumptions_ns = create_assumptions_namespace(assumptions)
        self.bast_renderer = BrianASTRenderer(variables)

    def render_node(self, node):
        '''
        Assumes that the node has already been fully processed by BrianASTRenderer
        '''
        node = super(ArithmeticSimplifier, self).render_node(node)
        # can't evaluate vector expressions, so abandon in this case
        if not node.scalar:
            return node
        # try fully evaluating using assumptions
        expr = NodeRenderer().render_node(node)
        val, evaluated = evaluate_expr(expr, self.assumptions_ns)
        if evaluated:
            if node.dtype=='boolean':
                val = bool(val)
                if hasattr(ast, 'NameConstant'):
                    newnode = ast.NameConstant(val)
                else:
                    # None is the expression context, we don't use it so we just set to None
                    newnode = ast.Name(repr(val), None)
            elif node.dtype=='integer':
                val = int(val)
            else:
                val = float(val)
            if node.dtype!='boolean':
                newnode = ast.Num(val)
            newnode.dtype = node.dtype
            newnode.scalar = True
            newnode.complexity = 0
            return newnode
        return node

    def render_BinOp(self, node):
        if node.dtype=='float': # only try to collect float type nodes
            if node.op.__class__.__name__ in ['Mult', 'Div', 'Add', 'Sub'] and not hasattr(node, 'collected'):
                newnode = self.bast_renderer.render_node(collect(node))
                newnode.collected = True
                return self.render_node(newnode)
        node.left = self.render_node(node.left)
        node.right = self.render_node(node.right)
        node = super(ArithmeticSimplifier, self).render_BinOp(node)
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
        # simplify e.g. 2*float to 2.0*float to make things more explicit: not strictly necessary
        # but might be useful for some codegen targets
        if node.dtype=='float' and op.__class__.__name__ in ['Mult', 'Add', 'Sub', 'Div']:
            for subnode in [node.left, node.right]:
                if subnode.__class__.__name__=='Num':
                    subnode.dtype = 'float'
                    subnode.n = float(subnode.n)
        return node


class Simplifier(BrianASTRenderer):
    def __init__(self, variables, scalar_statements):
        BrianASTRenderer.__init__(self, variables)
        self.loop_invariants = OrderedDict()
        self.loop_invariant_dtypes = {}
        self.n = 0
        self.node_renderer = NodeRenderer(use_vectorisation_idx=False)
        self.arithmetic_simplifier = ArithmeticSimplifier(variables)
        self.scalar_statements = scalar_statements

    def render_expr_with_additional_assumptions(self, expr, additional_assumptions):
        cur_arithmetic_simplifier = self.arithmetic_simplifier
        self.arithmetic_simplifier = ArithmeticSimplifier(self.variables,
                                                          assumptions=additional_assumptions)
        expr = self.render_expr(expr)
        self.arithmetic_simplifier = cur_arithmetic_simplifier
        return expr

    def render_expr(self, expr):
        node = brian_ast(expr, self.variables)
        node = self.arithmetic_simplifier.render_node(node)
        node = self.render_node(node)
        return self.node_renderer.render_node(node)

    def render_node(self, node):
        '''
        Assumes that the node has already been fully processed by BrianASTRenderer
        '''
        # can we pull this out?
        if node.scalar and node.complexity>0:
            expr = self.node_renderer.render_node(self.arithmetic_simplifier.render_node(node))
            if expr in self.loop_invariants:
                name = self.loop_invariants[expr]
            else:
                self.n += 1
                name = '_lio_'+str(self.n)
                self.loop_invariants[expr] = name
                self.loop_invariant_dtypes[name] = node.dtype
            # None is the expression context, we don't use it so we just set to None
            newnode = ast.Name(name, None)
            newnode.scalar = True
            newnode.dtype = node.dtype
            newnode.complexity = 0
            return newnode
        # otherwise, render node as usual
        return super(Simplifier, self).render_node(node)


def reduced_node(terms, op, curnode=None):
    for term in terms:
        if term is None:
            continue
        if curnode is None:
            curnode = term
        else:
            curnode = ast.BinOp(curnode, op(), term)
    return curnode


def cancel_identical_terms(primary, inverted):
    nr = NodeRenderer(use_vectorisation_idx=False)
    expressions = dict((node, nr.render_node(node)) for node in primary)
    expressions.update(**dict((node, nr.render_node(node)) for node in inverted))
    new_primary = []
    inverted_expressions = [expressions[term] for term in inverted]
    for term in primary:
        expr = expressions[term]
        if expr in inverted_expressions:
            new_inverted = []
            for iterm in inverted:
                if expressions[iterm]==expr:
                    expr = '' # handled
                else:
                    new_inverted.append(iterm)
            inverted = new_inverted
            inverted_expressions = [expressions[term] for term in inverted]
        else:
            new_primary.append(term)
    return new_primary, inverted


def collect(node):
    node.collected = True
    if node.__class__.__name__!='BinOp':
        return node
    terms_primary = []
    terms_inverted = []
    if node.op.__class__.__name__ in ['Mult', 'Div']:
        op_primary = ast.Mult
        op_inverted = ast.Div
        op_null = 1.0
        op_py_primary = lambda x, y: x*y
        op_py_inverted = lambda x, y: x/y
    elif node.op.__class__.__name__ in ['Add', 'Sub']:
        op_primary = ast.Add
        op_inverted = ast.Sub
        op_null = 0.0
        op_py_primary = lambda x, y: x+y
        op_py_inverted = lambda x, y: x-y
    else:
        return node
    collect_commutative(node, op_primary, op_inverted, terms_primary, terms_inverted)
    x = op_null
    remaining_terms_primary = []
    remaining_terms_inverted = []
    for term in terms_primary:
        if term.__class__.__name__=='Num':
            x = op_py_primary(x, term.n)
        else:
            remaining_terms_primary.append(term)
    for term in terms_inverted:
        if term.__class__.__name__=='Num':
            x = op_py_inverted(x, term.n)
        else:
            remaining_terms_inverted.append(term)
    if x!=op_null:
        num_node = ast.Num(x)
    else:
        num_node = None
    terms_primary = remaining_terms_primary
    terms_inverted = remaining_terms_inverted
    # final form that we want is:
    # ((num*prod(scalars)/prod(scalars))*prod(vectors))/prod(vectors)
    primary_scalar_terms = [term for term in terms_primary if term.scalar]
    inverted_scalar_terms = [term for term in terms_inverted if term.scalar]
    primary_scalar_terms, inverted_scalar_terms = cancel_identical_terms(primary_scalar_terms,
                                                                         inverted_scalar_terms)
    primary_vector_terms = [term for term in terms_primary if not term.scalar]
    inverted_vector_terms = [term for term in terms_inverted if not term.scalar]
    primary_vector_terms, inverted_vector_terms = cancel_identical_terms(primary_vector_terms,
                                                                         inverted_vector_terms)
    prod_primary_scalars = reduced_node(primary_scalar_terms, op_primary)
    prod_inverted_scalars = reduced_node(inverted_scalar_terms, op_primary)
    prod_primary_vectors = reduced_node(primary_vector_terms, op_primary)
    prod_inverted_vectors = reduced_node(inverted_vector_terms, op_primary)
    curnode = reduced_node([num_node, prod_primary_scalars], op_primary)
    if prod_inverted_scalars is not None:
        if curnode is None:
            curnode = ast.Num(float(op_null))
        curnode = ast.BinOp(curnode, op_inverted(), prod_inverted_scalars)
    curnode = reduced_node([curnode, prod_primary_vectors], op_primary)
    if prod_inverted_vectors is not None:
        if curnode is None:
            curnode = ast.Num(float(op_null))
        curnode = ast.BinOp(curnode, op_inverted(), prod_inverted_vectors)
    node = curnode
    if node is None: # everything cancelled
        node = ast.Num(float(op_null))
    node.collected = True
    return node


def collect_commutative(node, primary, inverted,
                        terms_primary, terms_inverted, add_to_inverted=False):
    op_primary = node.op.__class__ is primary
    # this should only be called with node a BinOp of type primary or inverted
    # left_exact is the condition that we can collect terms (we can do it with floats or add/sub,
    # but not integer mult/div)
    left_exact = (node.left.dtype=='float' or
                    (hasattr(node.left, 'op') and node.left.op.__class__.__name__ in ['Add', 'Sub']))
    if (node.left.__class__.__name__=='BinOp' and
            node.left.op.__class__ in [primary, inverted] and left_exact):
        collect_commutative(node.left, primary, inverted, terms_primary, terms_inverted,
                            add_to_inverted=add_to_inverted)
    else:
        if add_to_inverted:
            terms_inverted.append(node.left)
        else:
            terms_primary.append(node.left)
    right_exact = (node.right.dtype=='float' or
                    (hasattr(node.right, 'op') and node.right.op.__class__.__name__ in ['Add', 'Sub']))
    if (node.right.__class__.__name__=='BinOp' and
            node.right.op.__class__ in [primary, inverted] and right_exact):
        if node.op.__class__ is primary:
            collect_commutative(node.right, primary, inverted, terms_primary, terms_inverted,
                                add_to_inverted=add_to_inverted)
        else:
            collect_commutative(node.right, primary, inverted, terms_primary, terms_inverted,
                                add_to_inverted=not add_to_inverted)
    else:
        if (not add_to_inverted and op_primary) or (add_to_inverted and not op_primary):
            terms_primary.append(node.right)
        else:
            terms_inverted.append(node.right)
