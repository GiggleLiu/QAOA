from utils import get_bit

def max_cut_loss(z, clause_list):
    '''
    loss for a max cut problem.
    '''
    loss = 0
    for w, (start, end) in clause_list:
        loss -= w*(1-2*(get_bit(z, start)^get_bit(z, end)))
    return loss

def get_max_cut_clauses(graph):
    '''
    extract clauses for max-cut problem from a (weighted) graph: [-0.5Zi*Zj for i,j in edges].
    these clauses have 0.5 energy shift on each edge with repect to stardard definition.
    '''
    clause_list = []
    for edge in graph.es:
        weight = edge.attributes().get('weight',1)
        clause_list.append((-0.5*weight, (edge.source, edge.target)))
    return clause_list
