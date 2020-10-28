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

    def __init__(self,player, parent, parent_action,state,winner):

        self.parent = parent

        self.parent_action = parent_action

        self.player = player

        self.state = state

        self.winner = winner

    def terminal(self):
        if self.winner != 0:
            return True
        else:
            return False

    def reward(self):
        if self.winner == -1: #Draw
            return 0.0001

        elif self.winner == self.player:
            return 1

        else:
            return -1

    def __str__(self):
        return str(self.state.reshape((3,3)))

    def __hash__(self):
        "Nodes must be hashable"
        return hash((self.state.data.tobytes(), self.player, self.winner,self.parent_action))

    def __eq__(self, other):
        "Nodes must be comparable"
        return np.equal(other.state,self.state).all() and other.player == self.player and other.winner == self.winner \
                                        and other.parent_action == self.parent_action


class ZeroMCTS:
    """
    Search for AlphaGo Style Algorithms
    Modified slightly from vanilla MCTS
    """

    def __init__(self,env,nn,state_encoder,c=1.41,dir_noise=0.3,dir_eps=0.2):

        self._c = c
        self._dir_eps = dir_eps # How much of the dirichlet noise to incorporate
        self._dir_noise = dir_noise # dirichlet noise
        self._env = env
        self._nn = nn
        self._Qsa = defaultdict(float)
        self._Ns = defaultdict(int)
        self._Nsa = defaultdict(int)
        self._Ps = defaultdict(float)
        self._Vs = defaultdict(float)
        self._state_encoder = state_encoder # Function F: Node -> State

        self.root = None
        self.children = []
        self.dec_pts = []
        self.parents = []

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

        if parent not in self._Ps:
            self._Ps[parent] = prior_p

        a = child.parent_action
        self._Qsa[(parent,a)] = 0
        self._Nsa[(parent,a)] = 0

    def backup(self,path,r) -> None:

        leaf = path[-1]
        leaf_player = leaf.player

        #print("Running backup ")

        for node in reversed(path):
            self._Ns[node] += 1
            if not node.parent:
                continue

            self._Nsa[(node.parent,node.parent_action)] += 1
            if node.parent.player != leaf_player:
                #print("Node ....")
                #print(node)
                #print(f"Receiving reward -{r}")
                #print("Parent")
                #print(node.parent)
                #print("Parent action")
                #print(node.parent_action)
                self._Qsa[(node.parent,node.parent_action)] -= r
            else:
                #print("Node ....")
                #print(node)
                #print(f"Receiving reward +{r}")
                #print("Parent")
                #print(node.parent)
                #print("Parent action")
                #print(node.parent_action)
                self._Qsa[(node.parent,node.parent_action)] += r

    def expand(self,leaf: ZeroNode) -> None:
        """
        Expand the leaf node using the NN
        """
        #print("Running Expand")
        leaf_index = self.dec_pts.index(leaf)

        if len(self.children[leaf_index]) > 0 or leaf.terminal():
            return # The leaf must already be expanded

        children = self.get_children(leaf)

        p , v = self._nn.predict(self._state_encoder(leaf))

        if leaf not in self._Vs:
            self._Vs[leaf] = v

        for child in children:
            self.add_dec_pt(leaf,child,p)

    def get_children(self,node: ZeroNode) -> list:

        children = self.children[self.dec_pts.index(node)]

        if len(children) > 0:
            return children

        valid_actions = self._env.valid_actions(node.state)

        curr_board = node.state.copy()
        children = []
        player = 2 if node.player == 1 else 1
        for a in valid_actions:
            next_board = self._env.next_state(curr_board,a)
            winner = self._env.check_winner(next_board)
            next_node = ZeroNode(state=next_board, player=player, winner=winner, parent=node, parent_action=a)
            children.append(next_node)
        return children

    def policy(self,node,tau,max=False):
        """
        Give the current statistics in the tree
        return a distribution over actions for the
        current node.
        @param node:
        @param tau: temperature (float but generally 1)
        @return:
        """

        valid_actions = self._env.valid_actions(node.state)

        numerator = [self._Nsa[(node,a)]**tau for a in valid_actions]

        denom = sum(numerator)

        p = np.array(numerator) / denom

        p[np.isnan(p)] = 0

        p = p / p.sum()

        children = self.get_children(node)

        p_all = np.zeros((self._env.action_size,))

        if max:
            max_children = np.array(np.argwhere(p == np.max(p))).flatten()
            max_index = np.random.choice(max_children)
            p_all[max_index] = 1
            return children[max_index].parent_action,p_all

        p_all[valid_actions] = p

        try:
            child = np.random.choice(children, p=p)
        except:
            print("BROKEN")
        return child.parent_action , p_all

    def reset_tree(self):
        self._Qsa = defaultdict(float)
        self._Ns = defaultdict(int)
        self._Nsa = defaultdict(int)
        self._Psa = defaultdict(float)
        self._Vs = defaultdict(float)
        self.children = []
        self.dec_pts = []
        self.parents = []

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

        #print("Call Run!!!")
        #print("ROOT NODE")
        #print(node)

        if node in self.dec_pts:
            node_index = self.dec_pts.index(node)
            node = self.dec_pts[node_index] # Perfer to use the node that already exists

        path = self.search(node)

        leaf = path[-1]

        self.expand(leaf)

        reward = self.simulate(leaf)

        self.backup(path, reward)

    def search(self,node: ZeroNode) -> list:
        """
        Traverse tree from given node until leaf node is found
        """
        #print("Running Search \n")
        path = []
        while True:
            path.append(node)
            #print("Adding Node to path")
            #print(node)
            if node not in self.dec_pts:
                #print("Node not in pts returning")
                self._Ns[node] = 0 # just init hear will increment during backup
                return path

            node_index = self.dec_pts.index(node)

            if len(self.children[node_index]) == 0:
                #print("No children returning")
                return path

            for child in self.children[node_index]:

                if child in self.dec_pts:
                    continue

                else:
                    #print("Adding child node")
                    #print(child)
                    path.append(child)
                    return path

            node = self.select(node)

    def select(self,node: ZeroNode) -> ZeroNode:
        """
        @param node:
        @return:
        """
        #print("Selecting New Node")

        node_index = self.dec_pts.index(node)

        children = self.children[node_index]

        actions = self._env.valid_actions(node.state)

        if node in self._Ps:
            p = self._Ps[node]
        else:
            p, v = self._nn.predict(self._state_encoder(node))

            self._Ps[node] = p

            self._Vs[node] = v

        # renormalize
        p[[a for a in range(self._env.action_size) if a not in actions]] = 0

        p = p / p.sum()

        n_s = self._Ns[node]

        #print("Prior Prob")
        #print(p.reshape((3,3)))
        #print(f"Ns = {n_s}")

        if node == self.root:

            #print("Node is root adding noise")
            dir_noise = np.zeros(self._env.action_size)

            dir_noise[actions] = np.random.dirichlet([self._dir_noise] * len(actions))

            p = (1 - self._dir_eps) * p + self._dir_eps * dir_noise
            #print("New p")
            #print(p.reshape((3,3)))

        child_v = []
        for child in children:

            a = child.parent_action

            p_a = p[a]

            u_c = self._Qsa[(node,a)] / (self._Nsa[(node,a)] + 1) + self._c * p_a * (np.sqrt(n_s) / (1 + self._Nsa[(node,a)]))

            child_v.append((child, u_c))

        max_c = max(child_v, key=itemgetter(1))[0]

        #print("Found Max child")
        #print(max_c)
        return max_c

    def simulate(self, node) -> float:

        #print("Running simulate for Node...")
        #print(node)

        if node.terminal():
            #print("Node is terminal")
            r = node.reward()
            #print(f"Returning reward {r}")
            return r

        if node in self._Vs:
            r = self._Vs[node]
            #print(f"Node is in VS r={r}")

            return r
        else:
            p,v = self._nn.predict(self._state_encoder(node))

            self._Vs[node] = v

            #print(f"Node not in Vs new r = {v}")

            return v

    @property
    def points(self):
        return len(self.dec_pts)


