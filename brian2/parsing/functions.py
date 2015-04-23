import ast
import inspect

from brian2.utils.stringtools import deindent, indent, get_identifiers

from rendering import NodeRenderer

__all__ = ['AbstractCodeFunction',
           'abstract_code_from_function',
           'extract_abstract_code_functions',
           'substitute_abstract_code_functions',
           ]


class AbstractCodeFunction(object):
    '''
    The information defining an abstract code function
    
    Has attributes corresponding to initialisation parameters
    
    Parameters
    ----------
    
    name : str
        The function name.
    args : list of str
        The arguments to the function.
    code : str
        The abstract code string consisting of the body of the function less
        the return statement.
    return_expr : str or None
        The expression returned, or None if there is nothing returned.
    '''
    def __init__(self, name, args, code, return_expr):
        self.name = name
        self.args = args
        self.code = code
        self.return_expr = return_expr
    def __str__(self):
        s = 'def %s(%s):\n%s\n    return %s\n' % (self.name,
                                                  ', '.join(self.args),
                                                  indent(self.code),
                                                  self.return_expr)
        return s
    __repr__ = __str__


def abstract_code_from_function(func):
    '''
    Converts the body of the function to abstract code
    
    Parameters
    ----------
    func : function, str or ast.FunctionDef
        The function object to convert. Note that the arguments to the
        function are ignored.
        
    Returns
    -------
    func : AbstractCodeFunction
        The corresponding abstract code function
        
    Raises
    ------
    SyntaxError
        If unsupported features are used such as if statements or indexing.
    '''
    if callable(func):
        code = deindent(inspect.getsource(func))
        funcnode = ast.parse(code, mode='exec').body[0]
    elif isinstance(func, str):
        funcnode = ast.parse(func, mode='exec').body[0]
    elif func.__class__ is ast.FunctionDef:
        funcnode = func
    else:
        raise TypeError("Unsupported function type")
    
    if funcnode.args.vararg is not None:
        raise SyntaxError("No support for variable number of arguments")
    if funcnode.args.kwarg is not None:
        raise SyntaxError("No support for arbitrary keyword arguments")
    if len(funcnode.args.defaults):
        raise SyntaxError("No support for default values in functions")
    
    nodes = funcnode.body
    nr = NodeRenderer()
    lines = []
    return_expr = None
    for node in nodes:
        if node.__class__ is ast.Return:
            return_expr = nr.render_node(node.value)
            break
        else:
            lines.append(nr.render_node(node))
    abstract_code = '\n'.join(lines)
    try:
        # Python 2
        args = [arg.id for arg in funcnode.args.args]
    except AttributeError:
        # Python 3
        args = [arg.arg for arg in funcnode.args.args]
    name = funcnode.name
    return AbstractCodeFunction(name, args, abstract_code, return_expr)


def extract_abstract_code_functions(code):
    '''
    Returns a set of abstract code functions from function definitions.
    
    Returns all functions defined at the top level and ignores any other
    code in the string.
    
    Parameters
    ----------
    code : str
        The code string defining some functions.
        
    Returns
    -------
    funcs : dict
        A mapping ``(name, func)`` for ``func`` an `AbstractCodeFunction`.
    '''
    code = deindent(code)
    nodes = ast.parse(code, mode='exec').body
    funcs = {}
    for node in nodes:
        if node.__class__ is ast.FunctionDef:
            func = abstract_code_from_function(node)
            funcs[func.name] = func
    return funcs


class VarRewriter(ast.NodeTransformer):
    '''
    Rewrites all variable names in names by prepending pre
    '''
    def __init__(self, pre):
        self.pre = pre
    def visit_Name(self, node):
        return ast.Name(id=self.pre+node.id, ctx=node.ctx)
    def visit_Call(self, node):
        args = [self.visit(arg) for arg in node.args]
        return ast.Call(func=ast.Name(id=node.func.id, ctx=ast.Load()),
                        args=args, keywords=[], starargs=None, kwargs=None)


class FunctionRewriter(ast.NodeTransformer):
    '''
    Inlines a function call using temporary variables
    
    numcalls is the number of times the function rewriter has been called so
    far, this is used to make sure that when recursively inlining there is no
    name aliasing. The substitute_abstract_code_functions ensures that this is
    kept up to date between recursive runs.
    
    The pre attribute is the set of lines to be inserted above the currently
    being processed line, i.e. the inline code.
    
    The visit method returns the current line processed so that the function
    call is replaced with the output of the inlining.
    '''
    def __init__(self, func, numcalls=0):
        self.func = func
        self.numcalls = numcalls
        self.pre = []
        self.suspend = False
    def visit_Call(self, node):
        # we suspend operations during an inlining operation, then resume
        # afterwards, see below, so we only ever try to expand one inline
        # function call at a time, i.e. no f(f(x)). This case is handled
        # by the recursion.
        if self.suspend:
            return node
        # We only work with the function we're provided
        if node.func.id!=self.func.name:
            return node
        # Suspend while processing arguments (no recursion)
        self.suspend = True
        args = [self.visit(arg) for arg in node.args]
        self.suspend = False
        # The basename is used for function-local variables
        basename = '_inline_'+self.func.name+'_'+str(self.numcalls)
        # Assign all the function-local variables
        for argname, arg in zip(self.func.args, args):
            newpre = ast.Assign(targets=[ast.Name(id='%s_%s'%(basename, argname),
                                                  ctx=ast.Store())],
                                value=arg)
            self.pre.append(newpre)
        # Rewrite the lines of code of the function using the names defined
        # above
        vr = VarRewriter(basename+'_')
        for funcline in ast.parse(self.func.code).body:
            self.pre.append(vr.visit(funcline))
        # And rewrite the return expression
        return_expr = vr.visit(ast.parse(self.func.return_expr, mode='eval').body)
        self.pre.append(ast.Assign(targets=[ast.Name(id=basename,
                                                     ctx=ast.Store())],
                                   value=return_expr))
        # Finally we replace the function call with the output of the inlining
        newnode = ast.Name(id=basename)
        self.numcalls += 1
        return newnode


def substitute_abstract_code_functions(code, funcs):
    '''
    Performs inline substitution of all the functions in the code
    
    Parameters
    ----------
    code : str
        The abstract code to make inline substitutions into.
    funcs : list, dict or set of AbstractCodeFunction
        The function substitutions to use, note in the case of a dict, the
        keys are ignored and the function name is used.
        
    Returns
    -------
    code : str
        The code with inline substitutions performed.
    '''
    if isinstance(funcs, (list, set)):
        newfuncs = dict()
        for f in funcs:
            newfuncs[f.name] = f
        funcs = newfuncs
        
    code = deindent(code)
    lines = ast.parse(code, mode='exec').body

    # This is a slightly nasty hack, but basically we just check by looking at
    # the existing identifiers how many inline operations have already been
    # performed by previous calls to this function
    ids = get_identifiers(code)
    funcstarts = {}
    for func in funcs.values():
        subids = set([id for id in ids if id.startswith('_inline_'+func.name+'_')])
        subids = set([id.replace('_inline_'+func.name+'_', '') for id in subids])
        alli = []
        for subid in subids:
            p = subid.find('_')
            if p>0:
                subid = subid[:p]
            i = int(subid)
            alli.append(i)
        if len(alli)==0:
            i = 0
        else:
            i = max(alli)+1
        funcstarts[func.name] = i
    
    # Now we rewrite all the lines, replacing each line with a sequence of
    # lines performing the inlining
    newlines = []
    for line in lines:
        for func in funcs.values():
            rw = FunctionRewriter(func, funcstarts[func.name])
            line = rw.visit(line)
            newlines.extend(rw.pre)
            funcstarts[func.name] = rw.numcalls
        newlines.append(line)
        
    # Now we render to a code string
    nr = NodeRenderer()
    newcode = '\n'.join(nr.render_node(line) for line in newlines)
    
    # We recurse until no changes in the code to ensure that all functions
    # are expanded if one function refers to another, etc.
    if newcode==code:
        return newcode
    else:
        return substitute_abstract_code_functions(newcode, funcs)


if __name__=='__main__':
    if 1:
        def f(x):
            y = x*x
            return y
        def g(x):
            return f(x)+1
        code = '''
        z = f(x)
        z = f(x)+f(y)
        w = f(z)
        h = f(f(w))
        p = g(g(x))
        '''
        funcs = [abstract_code_from_function(f),
                 abstract_code_from_function(g),
                 ]
        print substitute_abstract_code_functions(code, funcs)
    if 0:
        code = '''
        def f(x):
            return x*x
        def g(V):
            V += 1
        '''
        funcs = extract_abstract_code_functions(code)
        for k, v in funcs.items():
            print v
    if 0:
        def f(V, w):
            V = w
            V += x
            x = y*z
            return x+y
        print abstract_code_from_function(f)
        