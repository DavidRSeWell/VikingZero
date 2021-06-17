"""
For generating some of the graphics that can be seen on the Arxiv paper
and the Blog pos associated with this project
"""
import numpy as np
import pydot
import yaml

from copy import deepcopy
from typing import AnyStr, Callable, List

from vikingzero.agents.alphago import AlphaZero
from vikingzero.agents.tictactoe_agents import TicTacToeMinMax,TicTacMCTSNode,TicTacToeMCTS
from vikingzero.environments.tictactoe_env import TicTacToe
from vikingzero.search import MCTS,ZeroMCTS, ZeroNode
from vikingzero.designer import DesignerZero

def uct(mcts,node,a):
    """
    Calculate the UCT value for a given node and action Q(s,a)
    """
    n_s = mcts.calculate_NS(node)
    p_a = mcts._Ps[node][a]
    return mcts._Qsa[(node,a)] / (mcts._Nsa[(node,a)] + 1) + mcts._c * p_a * (np.sqrt(n_s) / (1 + mcts._Nsa[(node,a)]))

def run_select(root,mcts):

    if root in mcts.dec_pts:
        node_index = mcts.dec_pts.index(root)
        root = mcts.dec_pts[node_index] # Perfer to use the node that already exists

    if root not in mcts.dec_pts:
        if root != mcts.root:
            parent_index = mcts.dec_pts.index(root.parent)
            mcts.children[parent_index].append(root)
            a = root.parent_action
            mcts._Qsa[(root.parent, a)] = 0
            mcts._Nsa[(root.parent, a)] = 0

        mcts.dec_pts.append(root)
        mcts.children.append([])


    path = mcts.search(root)

    return path

class Graphic:

    def __init__(self,config_path: AnyStr,img_dir: AnyStr):

        self.config_path = config_path
        self.img_dir = img_dir


    def add_mcts_node_label(self,g,node,mcts,node_i,parent_i):
        """
        Create label for the current node
        """
        label = "Q: {q} N: {nv} \n"
        if node.is_terminal():
            node_label = f" Leaf: Winner = {node.winner} \n"
        else:
            node_label = label.format(q=mcts._Q[node], nv=mcts._N[node])

        node_label += str(node)

        node_repr = f"node_{node_i}_parent_{parent_i}"

        g.add_node(pydot.Node(node_repr, label=node_label))

        return node_repr,node_label,g

    def alpha_mcts_png(self,mcts: ZeroMCTS,iter_num: int,step_num: int) -> None:
        """
        Create a png representation of a AlphZero Mcts tree
        :param mcts: ZeroMCTS Object object
        """
        g = pydot.Dot(graph_type="digraph")

        num_points = mcts.points
        for i in range(num_points):
            node = mcts.dec_pts[i]
            if node.terminal():
                node_label = f" Leaf: Winner = {node.winner} \n"
            else:
                v = mcts._Vs[node]
                node_label = f"Vs: {v:.3f} \n"

            node_label += str(node)
            if node.parent:
                g.add_node(pydot.Node(f"node_{i}_parent_{mcts.dec_pts.index(node.parent)}", label=node_label))
            else:
                g.add_node(pydot.Node(f"node_{i}_parent_None", label=node_label))

        for i in range(num_points):

            parent = mcts.dec_pts[i]
            for child in mcts.children[i]:
                uct_child = uct(mcts, parent, child.parent_action)
                c_index = mcts.dec_pts.index(child)
                p_c = np.round(mcts._Ps[parent].copy()[child.parent_action], 2)
                Qsa = mcts._Qsa[(parent, child.parent_action)]
                Nsa = mcts._Nsa[(parent, child.parent_action)]

                edge_label = f"action={child.parent_action} \n"
                edge_label += f"Qsa = {Qsa: .3f} \n"
                edge_label += f"Nsa = {Nsa} \n"
                edge_label += f"Ps = {p_c:.3f}\n"
                edge_label += f"UCT = {uct_child:.3f}"
                if parent.parent:
                    edge = pydot.Edge(f"node_{i}_parent_{mcts.dec_pts.index(parent.parent)}", f"node_{c_index}_parent_{mcts.dec_pts.index(parent)}", label=edge_label)
                else:
                    edge = pydot.Edge(f"node_{i}_parent_None", f"node_{c_index}_parent_{mcts.dec_pts.index(parent)}", label=edge_label)
                g.add_edge(edge)

        with open(self.img_dir + f"/image_{iter_num}_{step_num}.png", "wb") as img:
            img.write(g.create_png())

    def mcts_png(self,mcts: MCTS,iter_num: int,step_num: int) -> None:
        """
        Create a png representation of a AlphZero Mcts tree
        :param mcts: ZeroMCTS Object object
        """
        g = pydot.Dot(graph_type="digraph")

        node_i = 0
        root = mcts.root
        nodes = {root:(node_i,None,self.add_mcts_node_label(g,root,mcts,0,None)[0])}
        queue = [root]
        while len(queue) > 0:
            node = queue.pop()
            parent_i,_ , parent_label = nodes[node]
            if node not in mcts.children:
                continue
            for child in mcts.children[node]:
                node_i += 1
                repr,label,g = self.add_mcts_node_label(g,child,mcts,node_i,parent_i)
                edge = pydot.Edge(parent_label,repr,label="")
                g.add_edge(edge)
                nodes[child] = (node_i,parent_i,repr)
                queue.append(child)

        with open(self.img_dir + f"/mcts_image_{iter_num}_{step_num}.png", "wb") as img:
            img.write(g.create_png())

    def create_alpha_graphic(self,start_state: np.array,iters: int, save_iters: List = None):

        tictactoe = TicTacToe(start_board=start_state)

        if not save_iters:
            save_iters = [i for i in range(iters)]

        env = tictactoe

        root = ZeroNode(state=env.board, player=1, winner=0, parent=None, parent_action=None)

        with open(self.config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        designer = DesignerZero(env=env,agent_config=config["agent_config"],exp_config=config["exp_config"])

        agent_config = config["agent_config"]["agent1"]
        del agent_config["agent"]

        exp_config = config["exp_config"]

        alpha_agent = AlphaZero(env, **agent_config)

        zero_mcts = ZeroMCTS(env, alpha_agent._nn, alpha_agent.node_to_state, dir_eps=0)

        zero_mcts.children.append([])
        zero_mcts.dec_pts.append(root)
        zero_mcts.parents.append(None)
        zero_mcts.root = root

        for iter in range(iters):

            # STEP 1
            path = run_select(root,zero_mcts)

            if iter in save_iters:
                self.alpha_mcts_png(zero_mcts,iter_num=iter,step_num=1)

            leaf, a = path[-1]

            # STEP 2
            zero_mcts.expand(leaf)

            if iter in save_iters:
                self.alpha_mcts_png(zero_mcts,iter_num=iter,step_num=2)

            # Step 3
            reward = zero_mcts.simulate(leaf)

            if iter in save_iters:
                self.alpha_mcts_png(zero_mcts,iter_num=iter,step_num=3)

            # Step 4
            zero_mcts.backup(path, reward)

            if iter in save_iters:
                self.alpha_mcts_png(zero_mcts,iter_num=iter,step_num=4)

        designer.current_best = alpha_agent
        designer.current_player = deepcopy(alpha_agent)

        designer.train(alpha_agent,5)

        mem = alpha_agent.view_current_memory()

        return zero_mcts, alpha_agent

    def creat_mcts_graphic(self,start_state: np.array,iters: int,save_iters: List = None):

        tictactoe = TicTacToe(start_board=start_state)

        if not save_iters:
            save_iters = [i for i in range(iters)]

        env = tictactoe

        player = tictactoe.current_player

        root = TicTacMCTSNode(env,board=start_state,player=player)

        with open(self.config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        agent_config = config["agent_config"]["agent1"]
        del agent_config["agent"]

        mcts = TicTacToeMCTS(env,n_sim=50,c=1,player=player)

        mcts.root = root

        for iter in range(iters):

            path = mcts.select(root)

            #if iter in save_iters:
            #    self.mcts_png(mcts,iter_num=iter,step_num=1)

            leaf = path[-1]

            #if iter in save_iters:
            #    self.mcts_png(mcts,iter_num=iter,step_num=2)

            mcts.expand(leaf)

            if iter in save_iters:
                self.mcts_png(mcts,iter_num=iter,step_num=3)

            reward = mcts.simulate(leaf)

            mcts.backup(path, reward)

            #if iter in save_iters:
            #    self.mcts_png(mcts,iter_num=iter,step_num=4)

    def create_minimax_graphic(self,start_board: np.array):

        env = TicTacToe(start_board=start_board)

        env.current_player = 2

        minimax = TicTacToeMinMax(env,player=2)

        minimax.act(start_board)

        print("Done creating minimax graphic")


if __name__ == "__main__":

    config_path = "tests/tictactoe_alphago.yaml"

    img_path = "/Users/davidsewell/MLData/Thesis/MCTSVanilla"

    graphic = Graphic(config_path,img_path)

    start_board = np.zeros(9,)
    start_board[0] = 1
    start_board[3] = 2
    start_board[4] = 2
    start_board[6] = 1
    start_board[7] = 1
    start_board[8] = 2

    #graphic.create_alpha_graphic(start_board,iters=50,save_iters=[0,1,2,10,49])
    #graphic.create_minimax_graphic(start_board)
    graphic.creat_mcts_graphic(start_board,iters=50,save_iters=[0,10,49])