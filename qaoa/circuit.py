import numpy as np
import pdb
import scipy.sparse as sps
import scipy.sparse.linalg

class QAOACircuit(object):
    def __init__(self, num_bit, depth, cop, bop, v0):
        self.depth = depth
        self.num_bit = num_bit
        self.cop = cop
        self.bop = bop
        self.v0 = v0

    @property
    def num_param(self):
        '''number of parameters'''
        return 2*self.depth - 1

    def evolve(self, bs, gs):
        """
        constructs the final state |b, gamma> from the initial state
        by evolving a n-qubit quantum state with p applications of bop
        and (p-1) applications of cop, as per QAOA
        """
        v = self.v0
        for i in range(self.depth):
            v = sps.linalg.expm_multiply(-1.0j * bs[i] * self.bop, v)
            if i != self.depth-1:
                v = sps.linalg.expm_multiply(-1.0j * gs[i] * self.cop, v)
        return v

def build_qaoa_circuit(loss_table, num_bit, depth, v0=None, walk_mask=None):
    '''
    Args:
        loss_table (func): a table of loss used to construct C.
        num_bit (int): the number of bits.
        depth (int): the depth of circuit.
        v0 (1darray, default=|11...1>): initial state, represented in hilbert space.
        walk_mask (1darray, dtype=bool): mask out the desired space in B, to prevent from random walking out of desired space,
            default using the whole hilbert space.
    '''
    # default initial vector
    if v0 is None:
        v0 = np.zeros(2**num_bit)
        v0[-1] = 1

    # build evolution operators
    bop = b_op(num_bit, walk_mask)
    cop = c_op(loss_table)

    return QAOACircuit(num_bit, depth, cop, bop, v0)

def c_op(loss_table):
    """
    a 2^n by 2^n diagonal matrix with loss(z, graph) entries on the diagonal
    """
    return sps.diags(loss_table).tocsr()

def b_op(num_bit, walk_mask=None):
    """
    a 2^n by 2^n sparse matrix which allows a CTQW between
    quantum states representing valid vertex covers on the graph
    (which has exactly num_bit vertices)
    """
    N = 2**num_bit
    il, jl =[], []
    indices = np.arange(N)
    if walk_mask is not None:
        indices = indices[walk_mask]

    for x1 in indices:
        for j in range(num_bit):
            bit = 0x1 << j
            x2 = x1 ^ bit
            il.append(x1)
            jl.append(x2)
    return sps.coo_matrix((np.ones(len(il)), (il, jl)), shape=(N, N)).tocsr()
