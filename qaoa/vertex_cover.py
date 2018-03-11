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
    the objective function to minimize: - (counts the number of 0 bits in an integer),
    corresponding to maximising the number of vertices NOT in the vertex cover
    """
    if not mask[z]:
        return 0
    n = graph.vcount()
    s = 0
    for i in range(n):
        s += get_bit(z, i)
    return s - n
