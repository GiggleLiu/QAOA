'''
qaoa lib for training.
'''

import numpy as np

def get_qaoa_loss(circuit, loss_table):
    '''
    obtain the loss function for qaoa.

    Args:
        circuit (QAOACircuit): quantum circuit designed for QAOA.
        loss_table: a table of loss, iterating over 2**num_bit configurations.

    Returns:
        func, loss function with single parameter x.
    '''
    def loss(params):
        bs, gs = params[:circuit.depth], params[circuit.depth:]
        psi = circuit.evolve(bs, gs)
        pl = np.abs(psi)**2
        exp_val = (loss_table*pl).sum()
        print(exp_val)
        return exp_val
    return loss

def qaoa_result_digest(x, circuit, loss_table):
    """
    returns a quality from [0-1] of the result of the QAOA algorithm as compared to the optimal solution
    """
    num_bit, depth = circuit.num_bit, circuit.depth

    # get resulting distribution
    bs, gs = x[:depth], x[depth:]
    pl = np.abs(circuit.evolve(bs, gs))**2

    # calculate losses
    max_ind = np.argmax(pl)
    mean_loss = (loss_table*pl).sum()
    most_prob_loss = loss_table[max_ind]

    # get the exact solution
    exact_x = np.argmin(loss_table)
    exact_loss = loss_table[exact_x]

    print('Obtain: p(%d) = %.4f with loss = %.4f, mean loss = %.4f.'%(max_ind, pl[max_ind], most_prob_loss, mean_loss))
    print('Exact x = %d, loss = %.4f.'%(exact_x, exact_loss))
    return max_ind, most_prob_loss, exact_x, exact_loss
