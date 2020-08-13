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
        for node in reversed(path):
            self._N[node] += 1
            self._Q[node] += reward
            reward = 1 - reward  # 1 for me is 0 for my enemy, and vice versa

    def expand(self,node):
        """
        From the current node expand out one or
        more child nodes
        :param node:
        :return:
        """
        if node in self.children:
            return  # already expanded
        self.children[node] = node.get_children()

    def simulate(self,node) -> int:
        """
        Compute an estimate of the value fo the current node
        based on some simulate mechanism. Could be rollout or could be a NN
        :return:
        """
        player = node.player
        while True:
            if node.is_terminal():
                reward = node.reward(player)
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

        diff = parent.board - child.board
        diffs = np.where(diff != 0)[0]
        a = diffs[0]
        assert len(diffs) == 1
        #assert child.board[a] == parent.player
        return a

    def minmax_decision(self,node):
        """ Code borrowed from https://github.com/aimacode/aima-python/blob/master/games.py"""

        player = node.player
        #main_node = node

        def max_value(child_node):
            #print('------ max value -------')

            #print('------- BOARD ---------------')
            #print(child_node.board.reshape((3, 3)))

            if child_node.is_terminal():
                r = child_node.reward(player)
                #print(f'Current node has reward {r} for player {player}')
                return child_node.reward(player)

            v = -np.inf
            for child in child_node.get_children():
                child_val = min_value(child)
                #print(f'CHILD WITH VALUE {child_val}')
                #print(f'current value is {v}')
                v = max(v, child_val)
                #print(f'New value is {v}')
                #print('------- BOARD ---------------')
                #print(child.board.reshape((3, 3)))

            if v == -np.inf:
                print("stop")
            return v

        def min_value(child_node):
            #print('------ min value -------')
            #print('------- BOARD ---------------')
            #print(child_node.board.reshape((3, 3)))

            if child_node.is_terminal():
                r = child_node.reward(player)
                #print(f'Current node has reward {r} for player {player}')
                return child_node.reward(player)
            v = np.inf
            for child in child_node.get_children():
                child_val = max_value(child)
                #print(f'CHILD WITH VALUE {child_val}')
                #print(f'current value is {v}')
                v = min(v, child_val)
                #print(f'New value is {v}')
                #print('------- BOARD ---------------')
                #print(child.board.reshape((3, 3)))

            if v == np.inf:
                print("stop")
            return v

        # Body of minmax_decision:
        #children = node.get_children()
        #for c in children:
        #    if c not in self.children:
        #        self.children[main_node] = c
        #values = []
        #for child in node.get_children():
        #    values.append(min_value(child))
        #values = [min_value(child) for child in node.get_children()]
        return max(node.get_children(), key=lambda child: min_value(child))

    def run(self,node):
        test_board = np.zeros(9,)
        test_board[0] = 1
        test_board[1] = 1
        test_board[2] = 2
        test_board[4] = 2
        if np.equal(node.board,test_board).all():
            print("pause!")
        if node in self.children:
            return self.policy[node]

        else:
            max_child = self.minmax_decision(node)
            a = self.get_parent_action(node,max_child)
            self.children[node] = max_child
            node.action = a
            self.policy[node] = a
            return a


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
