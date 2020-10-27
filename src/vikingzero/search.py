"""
Abstract classes for MCTS and its associated nodes
Adapted from the below github account with minimal changes
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""

from abc import ABC,abstractmethod
from collections import defaultdict
from operator import itemgetter

import numpy as np


class MCTS:
    """
    Class to implement the main parts of Monte Carlo Tree search
    """
    def __init__(self,c):
        """
        :param c: Exploration hyper parameter
        """
        self._Q = defaultdict(int)
        self._N = defaultdict(int)

        self.c = c # exploratory parameter
        self.children = dict()

    def backup(self,path,reward):
        """
        Take the reward found from a leaf node and use it
        to adjust the values of the nodes in the path that was taken
        to get there.
        :param path: list of node object that were used in getting
        to the current leaf node
        :return:
        """
        current_player,r = reward
        for node in reversed(path):
            self._N[node] += 1
            if node.player != current_player: # this only works if losing is 0 loss
                self._Q[node] += r
            else:
                self._Q[node] -= r

    def expand(self,node):
        """
        From the current node expand out one or
        more child nodes
        :param node:
        :return:
        """
        if (node in self.children) or node.is_terminal():
            return  # already expanded
        self.children[node] = node.get_children()

    def simulate(self,node) -> tuple:
        """
        Compute an estimate of the value fo the current node
        based on some simulate mechanism. Could be rollout or could be a NN
        :return:
        """

        while True:
            if node.is_terminal():
                reward = node.reward()
                return (node.winner, reward)
            node = node.find_random_child()

    def select(self,node):
        """
        Select an action from the current node
        if not a leaf node
        :param node
        :return:
        """
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self.select_child(node)  # descend a layer deeper

    def select_child(self,node):
        """
        Provide mechanism for selecting child based off of current node.
        UCT is a common example
        :param node:
        :return:
        """
        raise NotImplementedError

    def run(self,node):
        """
        Run mcts simulations
        1: Select
        2: Expand
        3: Rollout
        4: Backup
        :param node:
        :return:
        """
        path = self.select(node)
        leaf = path[-1]
        self.expand(leaf)
        reward = self.simulate(leaf)
        self.backup(path, reward)


class MINIMAX:

    def __init__(self):
        """
        :param c: Exploration hyper parameter
        """

        self.children = dict()
        self.policy = dict()

    def get_parent_action(self,parent,child):

        a = parent.board
        b = child.board

        diff = a - b
        diffs = np.where(diff != 0)

        if a.shape[0] == 9: # tictactoe
            action = diffs[0][0]
        elif len(a.flatten()) == 42: # Connect4
            action = diffs[1][0]

        return action

    def minmax_decision(self,node):
        """ Code borrowed from https://github.com/aimacode/aima-python/blob/master/games.py"""

        player = node.player

        def max_value(child_node):

            if child_node.is_terminal():
                return child_node.reward(player)

            v = -np.inf
            for child in child_node.get_children():
                v = max(v, min_value(child))

            return v

        def min_value(child_node):

            if child_node.is_terminal():
                return child_node.reward(player)
            v = np.inf
            for child in child_node.get_children():

                v = min(v, max_value(child))

            return v

        values = []
        for child in node.get_children():
            values.append(min_value(child))

        v = [(child,min_value(child)) for child in node.get_children()]

        max_c = max(v,key=itemgetter(1))

        all_maxes = [child[0] for child in v if child[1] == max_c[1]]

        #max(node.get_children(), key=lambda child: min_value(child))

        return np.random.choice(all_maxes)

    def alpha_beta(self,node):

        player = node.player

        def max_value(child_node,alpha,beta):

            if child_node.is_terminal():
                return child_node.reward(player)

            v = -np.inf
            for child in child_node.get_children():
                v = max(v, min_value(child,alpha,beta))

                if v >= beta:
                    return v

                alpha = max(v,alpha)

            return v

        def min_value(child_node,alpha,beta):

            if child_node.is_terminal():
                return child_node.reward(player)
            v = np.inf
            for child in child_node.get_children():
                v = min(v, max_value(child,alpha,beta))

                if v <= alpha:
                    return v
                beta = min(v,beta)

            return v

        # Body of alpha_beta_search:
        best_score = -np.inf
        beta = np.inf
        best_action = None
        res = []
        for child in node.get_children():
            v = min_value(child, best_score, beta)
            #if v == best_score:
            #    best_action.append(child)
            #    continue

            res.append((child,v))
            continue


        max_c = max(res,key=itemgetter(1))

        all_maxes = [child[0] for child in res if child[1] == max_c[1]]

        return np.random.choice(all_maxes)

    def alpha_beta_depth(self,node,cutoff_test=None,n=5):

        player = node.player

        def max_value(child_node, alpha, beta, depth):

            if cutoff_test(child_node,depth):

                return self.simulate(child_node,player,n=n)

            v = -np.inf
            for child in child_node.get_children():

                v = max(v, min_value(child, alpha, beta, depth + 1))

                if v >= beta:
                    return v

                alpha = max(v, alpha)

            return v

        def min_value(child_node, alpha, beta, depth):

            if cutoff_test(child_node,depth):
                return self.simulate(child_node,player,n=n)

            v = np.inf
            for child in child_node.get_children():
                v = min(v, max_value(child, alpha, beta,depth + 1))

                if v <= alpha:
                    return v
                beta = min(v, beta)

            return v

        # Body of alpha_beta_search:
        best_score = -np.inf
        beta = np.inf
        best_action = None
        vs = []
        for child in node.get_children():
            v = min_value(child, best_score, beta,1)
            vs.append(v)
            if v > best_score:
                best_score = v
                best_action = child

        return best_action

    def run(self,node,type="minimax",d=2,n=5):

        max_child = None

        if type == "minimax":

            max_child = self.minmax_decision(node)

        elif type == "alphabeta":

            max_child = self.alpha_beta(node)

        elif type == "alphabeta_depth":

            if node in self.children:
                return self.policy[node]

            cutoff_test = lambda node, depth: depth > d or node.is_terminal()

            max_child = self.alpha_beta_depth(node,cutoff_test=cutoff_test,n=n)

        else:
            raise Exception(f"Inccorect minimax alogrithm type {type}")

        a = self.get_parent_action(node, max_child)

        self.children[node] = max_child

        node.action = a

        self.policy[node] = a

        return a

    def simulate(self,node,player,n=5) -> float:
        """
        Compute an estimate of the value fo the current node
        based on some simulate mechanism. Could be rollout or could be a NN
        :return:
        """

        r = 0
        for _ in range(n):
            curr_node = node
            while True:
                if curr_node.is_terminal():
                    r += curr_node.reward(player)
                    break
                curr_node = curr_node.find_random_child()
        return r


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def get_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self,player):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True


class ZeroNode:
    """
    Tree Node used for AlphaGo Agents
    """

    def __init__(self,index, player, parent, parent_action,state,terminal,r):
        super().__init__(Node)

        self.index = index

        self.parent = parent

        self.parent_action = parent_action

        self.player = player

        self.reward = r

        self.state = state

        self.terminal = terminal

    def __hash__(self):
        "Nodes must be hashable"
        return hash((self.state.data.tobytes(), self.player, self.winner,self.parent))

    def __eq__(self, other):
        "Nodes must be comparable"
        return np.equal(other.board,self.state).all() and other.player == self.player and other.winner == self.winner \
                and other.parent == self.parent and other.parent_action == self.parent_action


class ZeroMCTS:
    """
    Search for AlphaGo Style Algorithms
    Modified slightly from vanilla MCTS
    """

    def __init__(self,root: ZeroNode,env,nn,state_encoder,c=1.41,dir_noise=0.3,dir_eps=0.2,):

        self._c = c
        self._dir_eps = dir_eps # How much of the dirichlet noise to incorporate
        self._dir_noise = dir_noise # dirichlet noise
        self._env = env
        self._nn = nn
        self._Qsa = defaultdict(tuple)
        self._Ns = defaultdict(int)
        self._Nsa = defaultdict(tuple)
        self._Psa = defaultdict(float)
        self._Vs = defaultdict(float)
        self._state_encoder = state_encoder # Function F: Node -> State

        self.root = root
        self.children = [[]]
        self.dec_pts = [root]
        self.parents = [None]

    def add_dec_pt(self,parent,child,prior_p) -> None:
        """
        Add a decesion point to the tree
        initialize the child nodes with prior probability
        given by the policy network
        """
        self.dec_pts.append(child)
        self.children.append([])
        parent_index = self.dec_pts.index(parent)
        self.children[parent_index].append(child)

        if (parent,child.parent_action) not in self._Psa:
            a = child.parent_action
            self._Psa[(parent,a)] = prior_p
            self._Qsa[(parent,a)] = 0
            self._Nsa[(parent,a)] = 0

    def backup(self,path,r) -> None:

        leaf_player = path[-1].player

        for node in reversed(path):
            self._Ns[node] += 1
            self._Nsa[(node.parent,node.parent_action)] += 1
            if node.parent != leaf_player:
                self._Qsa[(node.parent,node.parent_action)] -= r
            else:
                self._Qsa[(node.parent,node.parent_action)] += r

    def expand(self,leaf: ZeroNode) -> None:
        """
        Expand the leaf node using the NN
        """
        if (leaf in self.dec_pts) or leaf.terminal:
            return # The leaf must already be expanded

        children = self.children[self.dec_pts.index(leaf)]

        p , v = self._nn.predict(self._state_encoder(leaf.state))

        if leaf not in self._Vs:
            self._Vs[leaf] = v

        for child in children:
            self.add_dec_pt(leaf,child,p[child.parent_action])

    def run(self, node: ZeroNode) -> None:
        """
            Run mcts simulations
            1: Select
            2: Expand
            3: Rollout
            4: Backup
            :param node:
            :return:
        """

        path = self.search(node)

        leaf = path[-1]

        self.expand(leaf)

        reward = self.simulate(leaf)

        self.backup(path, reward)

    def search(self,node: ZeroNode) -> list:
        """
        Traverse tree from given node until leaf node is found
        """
        path = []
        while True:
            path.append(node)
            if node not in self.dec_pts or not self.children[node.index]:
                return path

            for child in self.children[node.index]:
                if child in self.dec_pts:
                    continue

                else:
                    path.append(child)
                    return path

            node = self.select(node)

    def select(self,node: ZeroNode) -> ZeroNode:

        children = self.children[node.index]

        actions = self._env.valid_actions(node)

        if (node,node.parent_action) in self._Psa:
            p = self._Psa[(node,node.parent_action)]
        else:
            p, v = self._nn.predict(self._state_encoder(node.state))

            if node not in self._Vs:
                self._Vs[node] = v

        # renormalize
        p[[a for a in range(self._env.action_size) if a not in actions]] = 0

        p = p / p.sum()

        N_p = self._Ns[node]

        if node == self.root:

            dir_noise = np.zeros(self._env.action_size)

            dir_noise[actions] = np.random.dirichlet([self._dir_noise] * len(actions))

            p = (1 - self._dir_eps) * p + self._dir_eps * dir_noise

        child_v = []
        for child in children:

            a = child.parent_action

            p_a = p[a]

            if (child,a) not in self._Psa:
                self._Psa[(child,a)] = p_a

            u_c = self._Qsa[(node,a)] / (self._Nsa[(node,a)] + 1) + self._c * p_a * (np.sqrt(N_p) / (1 + self._Nsa[(child,a)]))

            child_v.append((child, u_c))

        return max(child_v, key=itemgetter(1))[0]

    def simulate(self, node) -> float:

        if node in self._Vs[node]:
            return self._Vs[node]

        else:
            p,v = self._nn.predict(self._state_encoder(node.state))

            self._Vs[node] = v

            return v



