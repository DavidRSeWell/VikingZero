
from abc import ABC,abstractmethod
from collections import defaultdict


class Node(ABC):

    def __init__(self,s):
        pass

    def get_children(self):
        pass


class MCTS(ABC):
    """
    Class to implement the main parts of Monte Carlo Tree search
    """
    def __init__(self):
        """
        :param c: Exploration hyper parameter
        """
        self._Q = defaultdict(int)
        self._N = defaultdict(int)

        self._c = c
        self._children = dict()


    @abstractmethod
    def backup(self,path,reward):
        """
        Take the reward found from a leaf node and use it
        to adjust the values of the nodes in the path that was taken
        to get there.
        :param path: list of node object that were used in getting
        to the current leaf node
        :return:
        """
        pass

    @abstractmethod
    def expand(self,node):
        """
        From the current node expand out one or
        more child nodes
        :param node:
        :return:
        """
        pass



    @abstractmethod
    def rollout(self,node):
        """
        Compute an estimate of the value fo the current node
        based on some rollout mechanism.
        :return:
        """
        pass

    @abstractmethod
    def select(self,node):
        """
        Select an action from the current node
        if not a leaf node
        :param node
        :return:
        """

    @abstractmethod
    def simulate(self,node):
        """
        Run mcts simulations
        1: Select
        2: Expand
        3: Rollout
        4: Backup
        :param node:
        :return:
        """
        pass
