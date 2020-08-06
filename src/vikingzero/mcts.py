"""
Abstract classes for MCTS and its associated nodes
Adapted from the below github account with minimal changes
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""

from abc import ABC,abstractmethod
from collections import defaultdict


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
        #invert_reward = True
        player = node.player
        while True:
            if node.is_terminal():
                reward = node.reward(player)
                return reward
                #return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            #invert_reward = not invert_reward

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
