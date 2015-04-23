from copy import copy

__all__ = ['topsort']

def topsort(graph):
    '''
    Topologically sort a graph
    
    The graph should be of the form ``{node: [list of nodes], ...}``.
    '''
    # make a copy so as not to destroy original
    graph = dict((k, copy(v)) for k, v in graph.iteritems())
    # Use the standard algorithm for topological sorting:
    # http://en.wikipedia.org/wiki/Topological_sorting
    # List that will contain the sorted elements
    sorted_items = []
    # set of all nodes with no incoming edges:
    no_incoming = set([node for node, edges in graph.iteritems() if len(edges)==0])

    while len(no_incoming):
        n = no_incoming.pop()
        sorted_items.append(n)
        # find nodes m with edges to n
        outgoing = [m for m, edges in graph.iteritems() if n in edges]
        for m in outgoing:
            graph[m].remove(n)
            if len(graph[m])==0:
                # no other dependencies
                no_incoming.add(m)
                
    if any([len(edges) > 0 for edges in graph.itervalues()]):
        raise ValueError('Cannot topologically sort cyclic graph.')
    
    return sorted_items


if __name__=='__main__':
    graph = {
        'a': ['b', 'c'],
        'b': ['c'],
        'c': [],
        }
    print topsort(graph)
    