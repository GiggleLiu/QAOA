import numpy as np
import pdb
from scipy.optimize import minimize, brute
import matplotlib.pyplot as plt

import circuit, qcircuit
from graph import show_graph
from train import get_qaoa_loss, qaoa_result_digest
from vertex_cover import is_vertex_cover, vertex_cover_loss, get_vertex_cover_clauses
from max_cut import max_cut_loss, get_max_cut_clauses

def solve_graph(graph, task, depth, runner, x0=None, optimizer='COBYLA', max_iter=1000):
    '''solve a problem defined by a graph.'''
    num_bit = graph.vcount()
    N = 2**num_bit

    if task == 'max-cut':
        clause_list = get_max_cut_clauses(graph)
        loss_func = lambda z: max_cut_loss(z, clause_list)
        valid_mask = None
    elif task == 'vertex-cover':
        valid_mask = [is_vertex_cover(graph, i) for i in range(N)]
        loss_func = lambda z: vertex_cover_loss(z, graph, valid_mask)
        if runner == 'projectq':
            print('warning, solving vertex_cover problem using projectq can not use `walk_mask`!')
            clause_list = get_vertex_cover_clauses(graph)
    else:
        raise

    loss_table = np.array([loss_func(z) for z in range(N)])

    if runner == 'scipy':
        cc = circuit.build_qaoa_circuit(loss_table, num_bit, depth, v0=None, walk_mask=valid_mask)
    elif runner == 'projectq':
        cc = qcircuit.build_qaoa_circuit(clause_list, num_bit, depth)
    else:
        raise

    # obtain and analyse results
    if x0 is None: x0 = np.zeros(cc.num_param)
    qaoa_loss, log = get_qaoa_loss(cc, loss_table) # the expectation value of loss function
    if optimizer == 'CMA-ES':
        import cma
        es = cma.CMAEvolutionStrategy(x0, 0.1*np.pi,
                                      inopts={'seed': np.random.randint(999999),
                                            })
        es.optimize(qaoa_loss, iterations=max_iter, verb_disp=1)
        best_x = es.best.x
    elif optimizer == 'COBYLA':
        best_x = minimize(qaoa_loss, x0=x0,
                method='COBYLA', options={'maxiter':max_iter}).x
    else:
        raise
    ans = qaoa_result_digest(best_x, cc, loss_table)
    #show_graph(graph, ans[2])


def solve_graph_greedy(graph, task, depth, runner, x0=None, optimizer='COBYLA', max_iter=1000, var_mask=None):
    '''solve a problem defined by a graph.'''
    num_bit = graph.vcount()
    N = 2**num_bit

    if task == 'max-cut':
        clause_list = get_max_cut_clauses(graph)
        loss_func = lambda z: max_cut_loss(z, clause_list)
        valid_mask = None
    elif task == 'vertex-cover':
        valid_mask = [is_vertex_cover(graph, i) for i in range(N)]
        loss_func = lambda z: vertex_cover_loss(z, graph, valid_mask)
        if runner == 'projectq':
            print('warning, solving vertex_cover problem using projectq can not use `walk_mask`!')
            clause_list = get_vertex_cover_clauses(graph)
    else:
        raise

    loss_table = np.array([loss_func(z) for z in range(N)])

    if runner == 'scipy':
        cc = circuit.build_qaoa_circuit(loss_table, num_bit, depth, v0=None, walk_mask=valid_mask)
    elif runner == 'projectq':
        cc = qcircuit.build_qaoa_circuit(clause_list, num_bit, depth)
    else:
        raise

    # obtain and analyse results
    if x0 is None: x0 = np.zeros(cc.num_param)
    var_mask = np.zeros(cc.num_param, dtype='bool')
    for i in range(cc.depth):
        var_mask[i] = True
        if i>0:
            var_mask[i+cc.depth-1] = True
        x = x0[var_mask]
        print('i=%d, mask=%s'%(i,var_mask))
        qaoa_loss, log = get_qaoa_loss(cc, loss_table, var_mask, x0) # the expectation value of loss function
        if optimizer == 'CMA-ES':
            import cma
            es = cma.CMAEvolutionStrategy(x, 0.1*np.pi,
                                          inopts={'seed': np.random.randint(999999),
                                                })
            es.optimize(qaoa_loss, iterations=max_iter, verb_disp=1)
            best_x = es.best.x
        elif optimizer == 'COBYLA':
            best_x = minimize(qaoa_loss, x0=x,
                    method='COBYLA', options={'maxiter':max_iter}).x
        elif optimizer == 'BRUTE':
            best_x = brute(qaoa_loss, ranges=[(0, np.pi)]+[(0, 2*np.pi)]*(i>0), Ns=10)
        else:
            raise
        x0[var_mask] = best_x
    plt.ion()
    plt.plot(log['loss'])
    pdb.set_trace()
    ans = qaoa_result_digest(x0, cc, loss_table)
    show_graph(graph, ans[2])


