import numpy as np


class RandomTicTacToeAgent:
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
        valid_actions = np.where(self._env.board == 0.0)[0]

        return np.random.choice(valid_actions,1)[0]
