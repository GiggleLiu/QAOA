'''
define problem using graph model.
'''

import igraph

from utils import get_bit

def random_graph(n, p, random_weight=False):
    '''
    a random graph for testing, frequently used for studying social network.
            # number of neighbor -> Poisson distribution.

    Args:
        n (int): number of vertices
        p (float): connection rate.

    Returns:
        igraph.Graph, graph model.
    '''
    graph = igraph.Graph.Erdos_Renyi(n=n, p=p)
    if random_weight:
        graph.es.set_attribute_values('weight', np.random.random(graph.ecount()))
    return graph

def show_graph(g, z):
    """
    opens external graph viewer
    """
    g.vs["color"] =  ['#ff0000' if get_bit(z, v) else '#ffffff' for v in range(g.vcount())]
    igraph.plot(g, vertex_label=[v.index for v in g.vs])


