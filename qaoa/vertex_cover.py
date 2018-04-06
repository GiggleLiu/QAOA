from utils import get_bit

def is_vertex_cover(graph, z):
    """
    checks if z (an integer) represents a valid vertex cover for graph adjacency
    matrix graph, with n vertices
    """
    for e in graph.es:
        if get_bit(z, e.source) == 0 and get_bit(z, e.target) == 0:
            return False
    return True

def vertex_cover_loss(z, graph, mask):
    """
    the objective function to minimize: -(# of 0 in a bit string),
    corresponding to maximising the number of vertices NOT in the vertex cover
    """
    if not mask[z]:
        return 0
    n = graph.vcount()
    s = 0
    for i in range(n):
        s += get_bit(z, i)
    return s - n

def get_vertex_cover_clauses(graph):
    '''
    C = \sum -0.5*(Zi+1), mapping is 0->down, 1->up.
    '''
    raise NotImplementedError()
    clause_list = []
    for v in graph.vs:
        clause_list.append(-(0.5, (v.index,)))
    return clause_list
