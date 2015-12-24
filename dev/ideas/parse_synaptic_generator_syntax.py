from brian2 import *
from brian2.parsing.rendering import NodeRenderer
import ast

def cname(obj):
    return obj.__class__.__name__


def parse_synapse_generator(expr):
    parse_error = ("Error parsing expression '%s'. Expression must have generator syntax, "
                   "for example 'k for k in range(i-10, i+10)'." % expr)
    supported_iterator_functions = ["range", "random_sample", "fixed_sample"]
    try:
        node = ast.parse('[%s]' % expr, mode='eval').body
    except Exception as e:
        raise SyntaxError(parse_error+" Error encountered was %s" % e)
    if cname(node)!='ListComp':
        raise SyntaxError(parse_error+" Expression is not a generator expression.")
    element = node.elt
    if len(node.generators)!=1:
        raise SyntaxError(parse_error+" Generator expression must involve only one iterator.")
    generator = node.generators[0]
    target = generator.target
    if cname(target)!='Name':
        raise SyntaxError(parse_error+" Generator must iterate over a single variable (not tuple, etc.).")
    iteration_variable = target.id
    iterator = generator.iter
    if cname(iterator)!='Call' or cname(iterator.func)!='Name':
        raise SyntaxError(parse_error+" Iterator expression must be one of the supported functions: "+
                          str(supported_iterator_functions))
    iterator_funcname = iterator.func.id
    if iterator_funcname not in supported_iterator_functions:
        raise SyntaxError(parse_error+" Iterator expression must be one of the supported functions: "+
                          str(supported_iterator_functions))
    if len(iterator.keywords) or iterator.kwargs is not None or iterator.starargs is not None:
        raise SyntaxError(parse_error+" Iterator expression must be one of the supported functions: "+
                          str(supported_iterator_functions)+". Only standard positional arguments supported.")
    iterator_args = iterator.args
    if len(generator.ifs)==0:
        condition = ast.parse('True', mode='eval').body
    elif len(generator.ifs)>1:
        raise SyntaxError(parse_error+" Generator must have at most one if statement.")
    else:
        condition = generator.ifs[0]
    nr = NodeRenderer()
    print
    print 'Original expression:', expr
    print '    Element:', nr.render_node(element)
    print '    Iteration variable:', iteration_variable
    print '    Iterator:', nr.render_node(iterator)
    print '    If:', nr.render_node(condition)


if __name__=='__main__':
    parse_synapse_generator('k for k in random_sample(0, N, p) if abs(i-k)<10')
    parse_synapse_generator('k+1 for k in range(i-100, i+100, 2)')
