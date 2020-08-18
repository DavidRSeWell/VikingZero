"""
Abstract classes for MCTS and its associated nodes
Adapted from the below github account with minimal changes
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""

from abc import ABC,abstractmethod
from collections import defaultdict

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
        winner,r = reward
        for node in reversed(path):
            self._N[node] += 1
            if node.player != winner:
                self._Q[node] += r
            #reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

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

    def simulate(self,node) -> int:
        """
        Compute an estimate of the value fo the current node
        based on some simulate mechanism. Could be rollout or could be a NN
        :return:
        """
        #player = node.player
        while True:
            if node.is_terminal():
                reward = node.reward()
                #return reward
                if not reward:
                    print('huh')
                return reward
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

        return max(node.get_children(), key=lambda child: min_value(child))

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
        for child in node.get_children():
            v = min_value(child, best_score, beta)
            if v > best_score:
                best_score = v
                best_action = child

        return best_action

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

        print(vs)
        return best_action

    def run(self,node,type="minimax",d=2,n=5):

        if node in self.children:
            return self.policy[node]

        else:

            max_child = None

            if type == "minimax":

                max_child = self.minmax_decision(node)

            elif type == "alphabeta":

                max_child = self.alpha_beta(node)

            elif type == "alphabeta_depth":

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
