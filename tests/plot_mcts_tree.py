from vikingzero.agents.alphago import CnnNNetSmall
from vikingzero.environments.tictactoe_env import TicTacToe
from vikingzero.search import ZeroMCTS, ZeroNode

import numpy as np
import pydot
import torch

from torch.autograd import Variable

tictactoe = TicTacToe()

env = tictactoe

player = env.check_turn(env.board)

winner = env.check_winner(env.board)

root = ZeroNode(state=env.board, player=player, winner=winner, parent=None, parent_action=None)


def _repr_png_(mcts):
    g = pydot.Dot(graph_type="digraph")

    num_points = mcts.points
    for i in range(num_points):
        node = mcts.dec_pts[i]
        if node.terminal():

            node_label = f" Leaf: Winner = {node.winner} "
        else:
            node_label = " Player: " + str(mcts.dec_pts[i].player)

        g.add_node(pydot.Node('node%d' % i, label=node_label))

    for i in range(num_points):

        parent = mcts.dec_pts[i]
        for child in mcts.children[i]:

            c_index = mcts.dec_pts.index(child)
            Qsa = mcts._Qsa[(parent,child.parent_action)]
            Nsa = mcts._Nsa[(parent,child.parent_action)]
            edge_label = f"action={child.parent_action} \n"
            edge_label += f"Qsa = {Qsa} \n"
            edge_label += f"Nsa = {Nsa}"

            edge = pydot.Edge('node%d' % i, 'node%d' % c_index,label=edge_label)
            g.add_edge(edge)

    write_path = 'tree.png'
    g.write(write_path, format="png")

    return g.create(g.prog, 'png')


def state_encoder(node):
    board = node.state
    board = board.reshape((3, 3))
    p1 = np.zeros(board.shape)
    p2 = np.zeros(board.shape)
    p = np.ones(board.shape)

    p1[board == 1] = 1
    p2[board == 2] = 1
    turn = env.check_turn(board)
    p = p if turn == 1 else p * -1
    board = np.stack((p1, p2, p))

    board = Variable(torch.from_numpy(board).type(dtype=torch.float))

    return board


nn = CnnNNetSmall(3, 3, 9, 32, 0.3)


zero_mcts = ZeroMCTS(env,nn,state_encoder,c=1.41,dir_noise=0.3,dir_eps=0.2)

zero_mcts.children.append([])
zero_mcts.dec_pts.append(root)
zero_mcts.parents.append(None)
zero_mcts.root = root

if __name__ == "__main__":

    for _ in range(30):
        zero_mcts.run(root)

    _repr_png_(zero_mcts)


    print("Done testing test_zero_node")




