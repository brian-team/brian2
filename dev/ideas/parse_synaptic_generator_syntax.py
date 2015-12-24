from brian2 import *
from brian2.parsing.rendering import NodeRenderer
import ast

def parse_synapse_generator(expr):
    node = ast.parse('[%s]' % expr, mode='eval').body
    nr = NodeRenderer()
    print 'Element:', nr.render_node(node.elt)
    print 'Variable:', node.generators[0].target.id
    print 'Variable in:', nr.render_node(node.generators[0].iter)
    if len(node.generators[0].ifs)==1:
        print 'If:', nr.render_node(node.generators[0].ifs[0])
    elif len(node.generators[0].ifs)>1:
        raise SyntaxError("Only allowed one if statement")

parse_synapse_generator('k+1 for k in range(i-100, i+100, 2) if f(k)')
