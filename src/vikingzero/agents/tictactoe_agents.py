import numpy as np

from ..environments.tictactoe_env import TicTacToe
from ..search import MCTS,MINIMAX,Node


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

    def act(self,s,r=0):
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

        s = TicTacMiniNode(self._env,board,self._player)

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

    def reward(self):
        if self.winner == -1:
            #TODO
            return 0.5
        elif int(self.winner) != self.player:
            reward = 1
            return reward
        else:
            print("wtf")

    @property
    def actions(self):
        return self.env.actions(self.board)

    def __eq__(self, other):
        return np.equal(other.board,self.board).all() and other.player == self.player and other.winner == self.winner

    def __hash__(self):
        return hash((self.board.data.tobytes(),self.player,self.winner))


class TicTacMiniNode(Node):
    def __init__(self,env: TicTacToe,board: np.array, player: int, winner = 0):
        self.action = None
        self.board = board
        self.player = player
        self.env = env
        self.winner = winner

    def find_random_child(self):
        children = self.get_children()
        random_child = np.random.choice(children, 1)[0]
        assert type(random_child) == TicTacMiniNode
        return random_child

    def get_children(self):

        valid_actions = self.env.actions(self.board)
        curr_board = self.board.copy()
        children = []
        player = 2 if self.player == 1 else 1
        for a in valid_actions:
            next_board = self.env.next_state(curr_board,a, self.player)
            r, winner = self.env.check_winner(next_board)
            next_node = TicTacMiniNode(env=self.env, board=next_board, player=player, winner=winner)
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


class TicTacToeMCTS(MCTS):

    def __init__(self,env: TicTacToe, n_sim: int = 50,c: int = 1,player = 1):
        super().__init__(c)

        print(f"Loaded MCTS with nsim = {n_sim} c = {c} player = {player}")
        self._env = env
        self._num_sim = n_sim
        self._player = player

    def act(self,board,render=False):

        s = TicTacToeNode(self._env,board,self._player,0)
        # First rum simulations to collect
        # Tree statistics
        for _ in range(self._num_sim):

            self.run(s)

        a = self.get_max_action(s,render)

        return a

    def create_value_board(self,parent,children):

        board = parent.board.copy()
        board = np.ones(board.shape)*-np.inf
        board2 = np.ones(board.shape)*-np.inf
        for c in children[parent]:
            v = self.value(c)
            a = self.get_parent_ation(parent,c)
            board[a] = self._Q[c]
            board2[a] = self._N[c]

        return board,board2

    def get_max_action(self,s,render=False) -> int:

        if s not in self.children:
            raise Exception(f"Node {s} does not exist in the tree")

        if self._env.is_win(s.board):
            raise Exception("Attempting to compute action on a leaf node")

        children = self.children

        values = [self.value(c) for c in children[s]]

        if render:

            print('------- CURRENT BOARD -------')
            print(s.board.reshape((3,3)))

            print('-------- VALUES of BOARD --------')
            v_board,n_board = self.create_value_board(s,children)
            print(v_board.reshape((3,3)))

            print('-------- COUNTS ----------------')
            print(n_board.reshape((3,3)))

        max_child = children[s][np.argmax(np.array(values))]

        action = self.get_parent_ation(s,max_child)

        return action

    def get_parent_ation(self,parent,child):

        a = parent.board

        b = child.board

        diff = a - b
        diffs = np.where(diff != 0)

        action = diffs[0][0]

        return action

    def select_child(self,node):

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = np.log(self._N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self._Q[n] / self._N[n] + self.c * np.sqrt(
                log_N_vertex / self._N[n]
            )

        return max(self.children[node], key=uct)

    def value(self,s):

        if self._N[s] == 0:
            return 0 #TODO return 0 or -inf?

        return self._Q[s] / self._N[s]

