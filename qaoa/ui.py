#!/usr/bin/env python
import numpy as np
import pdb
import fire

from views import solve_graph_greedy, solve_graph
from graph import random_graph

class UI(object):
    def vc(self):
        '''minimum vertex-cover problem'''
        import random
        random.seed(2)
        np.random.seed(2)
        num_bit, prob = 8, 0.3  # 30% of edges are connected.
        p=3
        graph = random_graph(num_bit, prob)
        solve_graph(graph, task='vertex-cover', runner='scipy', depth=p, x0=np.random.random(2*p-1))

    def mc(self, runner='scipy'):
        '''max-cut problem, runner = ('scipy'|'projectq')'''
        import random
        random.seed(2)
        np.random.seed(2)
        num_bit, prob = 8, 0.3  # 30% of edges are connected.

        # define loss function and circuit
        graph = random_graph(num_bit, prob, random_weight=False)
        solve_graph(graph, task='max-cut', runner=runner, depth=10, optimizer='COBYLA', max_iter=200)
        #solve_graph_greedy(graph, task='max-cut', runner=runner, depth=10, optimizer='COBYLA', max_iter=100)

if __name__ == "__main__":
    fire.Fire(UI)
