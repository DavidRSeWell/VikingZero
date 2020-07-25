import numpy as np

from ..environments.connect4_env import Connect4
from ..mcts import MCTS,Node


class RandomConnect4Agent:
    """
    Agent that acts randomly while playing Connect4
    """
    def __init__(self,env):

        self._env = env

    def act(self,s):
        """
        Take an action given a state
        :param s:
        :return:
        """
        valid_actions = self._env.valid_actions(s)

        return np.random.choice(valid_actions,1)[0]


class Connect4MCTS(MCTS):

    def __init__(self,env: Connect4, n_sim: int):
        super().__init__()
        self._env = env
        self._num_sim = n_sim

    def act(self,s):

        # First rum simulations to collect
        # Tree statistics
        for _ in range(self._num_sim):

            self.simulate(s)

        a = self.get_max_action(s)


    def create_node(self,s):
        pass


    def get_max_action(self,s) -> Node:

        if s not in self._children:
            raise Exception(f'Node {s} does not exist in the tree')

        if self._env.check_winner(s.render()):
            raise Exception('Attempting to compute action on a leaf node')

        children = s.children

        return Node

    def value(self,s):

        if self._N[s] == 0:
            return 0 #TODO return 0 or -inf?

        return self._Q[s] / self._N[s]
