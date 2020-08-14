import numpy as np

from ..environments.tictactoe_env import TicTacToe
from ..search import Node,MINIMAX

class HumanAgent:

    def __init__(self,env):
        self._env = env

    def act(self,s):

        while True:
            try:
                action = int(input("Enter Action: "))


                valid_actions = self._env.actions(self._env.board)

                if action in valid_actions:
                    return action
            except:

                print(f"action is invalid valid action are")

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

class TicTacToeMinMax(MINIMAX):

    def __init__(self,env: TicTacToe,player = 1,type="minimax"):
        super().__init__()

        self._env = env
        self._player = player
        self._type = type

    def act(self,board):

        s = TicTacToeNode(self._env,board,self._player)

        a = self.run(s,type=self._type)

        return a

class TicTacToeNode(Node):

    def __init__(self,env: TicTacToe,board: np.array, player: int, winner = 0):
        self.action = None
        self.board = board
        self.player = player
        self.env = env
        self.winner = winner

    def find_random_child(self):
        children = self.get_children()
        random_child = np.random.choice(children, 1)[0]
        assert type(random_child) == TicTacToeNode
        return random_child

    def get_children(self):

        valid_actions = self.env.actions(self.board)
        curr_board = self.board.copy()
        children = []
        player = 2 if self.player == 1 else 1
        for a in valid_actions:
            next_board = self.env.next_state(curr_board,a, self.player)
            r, winner = self.env.check_winner(next_board)
            next_node = TicTacToeNode(env=self.env, board=next_board, player=player, winner=winner)
            children.append(next_node)
        return children

    def is_terminal(self):
        return True if self.env.is_win(self.board) else False

    def next_state(self,a):
        return self.env.next_state(self.board,a,self.player)

    def reward(self,player):
        reward = 1
        if self.winner == -1:
            return 0
        elif int(self.winner) != player:
            reward = -1
        return reward

    @property
    def actions(self):
        return self.env.actions(self.board)

    def __eq__(self, other):
        return np.equal(other.board,self.board).all() and other.player == self.player and other.winner == self.winner

    def __hash__(self):
        return hash((self.board.data.tobytes(),self.player,self.winner))

