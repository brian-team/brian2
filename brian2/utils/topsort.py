from copy import copy

__all__ = ["topsort"]


def topsort(graph):
    """
    Topologically sort a graph

    The graph should be of the form ``{node: [list of nodes], ...}``.
    """
    try:
        from graphlib import TopologicalSorter

        sorter = TopologicalSorter(graph)
        return list(sorter.static_order())
    except ImportError:
        # TODO: Can be removed when we depend on Python >= 3.9
        # make a copy so as not to destroy original
        graph = dict((k, copy(v)) for k, v in graph.items())
        # Use the standard algorithm for topological sorting:
        # http://en.wikipedia.org/wiki/Topological_sorting
        # List that will contain the sorted elements
        sorted_items = []
        # set of all nodes with no incoming edges:
        no_incoming = {node for node, edges in graph.items() if len(edges) == 0}

        while len(no_incoming):
            n = no_incoming.pop()
            sorted_items.append(n)
            # find nodes m with edges to n
            outgoing = [m for m, edges in graph.items() if n in edges]
            for m in outgoing:
                graph[m].remove(n)
                if len(graph[m]) == 0:
                    # no other dependencies
                    no_incoming.add(m)

        if any([len(edges) > 0 for edges in graph.values()]):
            raise ValueError("Cannot topologically sort cyclic graph.")

        return sorted_items
