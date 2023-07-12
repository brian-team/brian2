from graphlib import TopologicalSorter

__all__ = ["topsort"]


def topsort(graph):
    """
    Topologically sort a graph

    The graph should be of the form ``{node: [list of nodes], ...}``.

    Uses `graphlib.TopologicalSorter`.
    """
    sorter = TopologicalSorter(graph)
    return list(sorter.static_order())
