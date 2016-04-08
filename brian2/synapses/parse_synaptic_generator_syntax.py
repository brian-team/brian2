from brian2 import *
from brian2.parsing.rendering import NodeRenderer
import ast

__all__ = ['parse_synapse_generator']


def _cname(obj):
    return obj.__class__.__name__


def handle_range(*args, **kwds):
    '''
    Checks the arguments/keywords for the range iterator

    Should have 1-3 positional arguments.

    Returns a dict with keys low, high, step. Default values are
    low=0, step=1.
    '''
    if len(args) == 0 or len(args) > 3:
        raise SyntaxError("Range iterator takes 1-3 positional arguments.")
    if len(kwds):
        raise SyntaxError("Range iterator doesn't accept any keyword "
                          "arguments.")
    if len(args) == 1:
        high = args[0]
        low = '0'
        step = '1'
    elif len(args) == 2:
        low, high = args
        step = '1'
    else:
        low, high, step = args
    return {'low': low, 'high': high, 'step': step}


def handle_sample(*args, **kwds):
    '''
    Checks the arguments/keywords for the sample iterator

    Should have 1-3 positional arguments and 1 keyword argument (either p or
    size).

    Returns a dict with keys ``low, high, step, sample_size, p, size``. Default
    values are ``low=0``, ``step=1`. Sample size will be either ``'random'`` or
    ``'fixed'``. In the first case, ``p`` will have a value and size will be
    ``None`` (and vice versa for the second case).
    '''
    if len(args) == 0 or len(args) > 3:
        raise SyntaxError("Sample iterator takes 1-3 positional arguments.")
    if len(kwds) != 1 or ('p' not in kwds and 'size' not in kwds):
        raise SyntaxError("Sample iterator accepts one keyword argument, "
                          "either 'p' or 'size'.")
    if len(args) == 1:
        high = args[0]
        low = '0'
        step = '1'
    elif len(args) == 2:
        low, high = args
        step = '1'
    else:
        low, high, step = args
    if 'p' in kwds:
        sample_size = 'random'
        p = kwds['p']
        size = None
    else:
        sample_size = 'fixed'
        size = kwds['size']
        p = None
    return {'low': low, 'high': high, 'step': step,
            'p': p, 'size': size, 'sample_size': sample_size}


iterator_function_handlers = {
    'range': handle_range,
    'sample': handle_sample,
    }


def parse_synapse_generator(expr):
    '''
    Returns a parsed form of a synapse generator expression.

    The general form is:

    ``element for iteration_variable in iterator_func(...)``

    or

    ``element for iteration_variable in iterator_func(...) if if_expression``

    Returns a dictionary with keys:

    ``original_expression``
        The original expression as a string.
    ``element``
        As above, a string expression.
    ``iteration_variable``
        A variable name, as above.
    ``iterator_func``
        String. Either ``range`` or ``sample``.
    ``if_expression``
        String expression or ``None``.
    ``iterator_kwds``
        Dictionary of key/value pairs representing the keywords. See
        `handle_range` and `handle_sample`.
    '''
    nr = NodeRenderer(use_vectorisation_idx=False)
    parse_error = ("Error parsing expression '%s'. Expression must have "
                   "generator syntax, for example 'k for k in range(i-10, "
                   "i+10)'." % expr)
    try:
        node = ast.parse('[%s]' % expr, mode='eval').body
    except Exception as e:
        raise SyntaxError(parse_error + " Error encountered was %s" % e)
    if _cname(node) != 'ListComp':
        raise SyntaxError(parse_error + " Expression is not a generator "
                                        "expression.")
    element = node.elt
    if len(node.generators) != 1:
        raise SyntaxError(parse_error + " Generator expression must involve "
                                        "only one iterator.")
    generator = node.generators[0]
    target = generator.target
    if _cname(target) != 'Name':
        raise SyntaxError(parse_error + " Generator must iterate over a single "
                                        "variable (not tuple, etc.).")
    iteration_variable = target.id
    iterator = generator.iter
    if _cname(iterator) != 'Call' or _cname(iterator.func) !=  'Name':
        raise SyntaxError(parse_error + " Iterator expression must be one of "
                                        "the supported functions: " +
                          str(iterator_function_handlers.keys()))
    iterator_funcname = iterator.func.id
    if iterator_funcname not in iterator_function_handlers:
        raise SyntaxError(parse_error + " Iterator expression must be one of "
                                        "the supported functions: " +
                          str(iterator_function_handlers.keys()))
    if (getattr(iterator, 'starargs', None) is not None or
                getattr(iterator, 'kwargs', None) is not None):
        raise SyntaxError(parse_error + " Star arguments not supported.")
    args = []
    for argnode in iterator.args:
        args.append(nr.render_node(argnode))
    keywords = {}
    for kwdnode in iterator.keywords:
        keywords[kwdnode.arg] = nr.render_node(kwdnode.value)
    try:
        iterator_handler = iterator_function_handlers[iterator_funcname]
        iterator_kwds = iterator_handler(*args,**keywords)
    except SyntaxError as exc:
        raise SyntaxError(parse_error + " " + exc.msg)
    if len(generator.ifs) == 0:
        condition = ast.parse('True', mode='eval').body
    elif len(generator.ifs) > 1:
        raise SyntaxError(parse_error + " Generator must have at most one if "
                                        "statement.")
    else:
        condition = generator.ifs[0]
    parsed = {
        'original_expression': expr,
        'element': nr.render_node(element),
        'iteration_variable': iteration_variable,
        'iterator_func': iterator_funcname,
        'iterator_kwds': iterator_kwds,
        'if_expression': nr.render_node(condition),
        }
    return parsed


if __name__=='__main__':
    for parsed in [
                parse_synapse_generator('k for k in sample(0, N, p=p) if abs(i-k)<10'),
                parse_synapse_generator('k for k in sample(0, N, size=5) if abs(i-k)<10'),
                parse_synapse_generator('k+1 for k in range(i-100, i+100, 2)'),
                ]:
        print 'PARSED:'
        for k, v in parsed.items():
            print '    '+k+': '+str(v)
