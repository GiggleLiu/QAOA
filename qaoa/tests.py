from scipy.optimize import minimize
import numpy as np
import igraph
import pdb
from profilehooks import profile

import circuit, qcircuit
from train import get_qaoa_loss, qaoa_result_digest
from utils import get_bit
from vertex_cover import is_vertex_cover, vertex_cover_loss
from max_cut import max_cut_loss, get_max_cut_clauses

def show_graph(g, z):
    """
    opens external graph viewer
    """
    g.vs["color"] =  ['#ff0000' if get_bit(z, v) else '#ffffff' for v in range(g.vcount())]
    igraph.plot(g, vertex_label=[v.index for v in g.vs])

#@profile
def test_sps_vc():
    # a random graph for testing, frequently used for studying social network.
    # number of neighbor -> Poisson distribution.
    import random
    random.seed(5)
    num_bit, prob = 13, 0.5  # 30% of edges are connected.
    depth = 2

    # define loss function and circuit
    graph = igraph.Graph.Erdos_Renyi(n=num_bit, p=prob)
    valid_mask = [is_vertex_cover(graph, i) for i in range(2**num_bit)]
    loss_table = np.array([vertex_cover_loss(z, graph, valid_mask) for z in range(2**num_bit)])

    cc = circuit.build_qaoa_circuit(loss_table, num_bit, depth, v0=None, walk_mask=valid_mask)

    # obtain and analyse results
    x0 = np.zeros(cc.num_param)
    result = minimize(get_qaoa_loss(cc, loss_table), x0=x0, method='COBYLA', options={'maxiter':1000})
    ans = qaoa_result_digest(result.x, cc, loss_table)
    show_graph(graph, ans[0])

def test_sps_mc():
    # a random graph for testing, frequently used for studying social network.
    # number of neighbor -> Poisson distribution.
    import random
    random.seed(2)
    np.random.seed(2)
    num_bit, prob = 10, 0.3  # 30% of edges are connected.
    depth = 2

    # define loss function and circuit
    graph = igraph.Graph.Erdos_Renyi(n=num_bit, p=prob)
    #graph.es.set_attribute_values('weight', np.random.random(graph.ecount()))
    clause_list = get_max_cut_clauses(graph)
    loss_table = np.array([max_cut_loss(z, clause_list) for z in range(2**num_bit)])

    cc = circuit.build_qaoa_circuit(loss_table, num_bit, depth, v0=None)

    # obtain and analyse results
    x0 = np.zeros(cc.num_param)
    result = minimize(get_qaoa_loss(cc, loss_table), x0=x0,
            method='COBYLA', options={'maxiter':1000})
    ans = qaoa_result_digest(result.x, cc, loss_table)
    show_graph(graph, ans[2])


def test_q_mc():
    import random
    random.seed(2)
    np.random.seed(2)
    num_bit, prob = 10, 0.3  # 30% of edges are connected.
    depth = 2

    # define loss function and circuit
    graph = igraph.Graph.Erdos_Renyi(n=num_bit, p=prob)
    #graph.es.set_attribute_values('weight', np.random.random(graph.ecount()))
    clause_list = get_max_cut_clauses(graph)
    loss_table = np.array([max_cut_loss(z, clause_list) for z in range(2**num_bit)])

    cc = qcircuit.build_qaoa_circuit(clause_list, num_bit, depth)

    # obtain and analyse results
    x0 = np.zeros(cc.num_param)
    result = minimize(get_qaoa_loss(cc, loss_table), x0=x0, method='COBYLA', options={'maxiter':1000})
    ans = qaoa_result_digest(result.x, cc, loss_table)
    show_graph(graph, ans[2])


if __name__ == "__main__":
    #test_sps_vc()
    #test_sps_mc()
    test_q_mc()
