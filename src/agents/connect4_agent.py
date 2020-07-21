import numpy as np


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
