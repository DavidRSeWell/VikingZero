import numpy as np
import torch

from abc import abstractmethod
from collections import namedtuple

from ..environments.connect4_env import Connect4
from ..search import MCTS,MINIMAX,Node
from ..connect4_uci_test import Net as UCINet

_CN4 = namedtuple("Connect4Node", "env board player winner")


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


class Connect4MinMax(MINIMAX):

    def __init__(self,env: Connect4,player = 1,type="minimax",depth=2,n_sims=1):
        super().__init__()

        print(f"Loading MINMAX agent with type = {type}")
        self._env = env
        self._d = depth
        self._n = n_sims
        self._player = player
        self._type = type

    def act(self,board):

        s = Connect4MiniNode(self._env,board,self._player)

        a = self.run(s,type=self._type,d=self._d,n=self._n)

        return a


class Connect4Node(Node):
    """
    Base class. Dont use.
    """
    def __init__(self,env: Connect4,node_type,board: np.array, player: int, winner = 0):

        self.board = board
        self.env = env
        self.node = node_type
        self.player = player
        self.winner = winner

    def get_children(self):
        "All possible successors of this board state"

        valid_actions = self.env.valid_actions(self.board)
        curr_board = self.board.copy()
        children = []
        player = 2 if self.player == 1 else 1
        for a in valid_actions:
            board_row = self.env.process_move(curr_board, a)
            next_board = curr_board.copy()
            next_board[board_row,a] = self.player
            winner = self.env.check_winner(next_board)
            next_node = self.node(env=self.env,board=next_board,player=player,winner=winner)
            children.append(next_node)
        return children

    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        children = self.get_children()
        random_child = np.random.choice(children, 1)[0]
        assert type(random_child) == self.node
        return random_child

    def is_terminal(self):
        "Returns True if the node has no children"
        return True if self.env.check_winner(self.board) else False

    @abstractmethod
    def reward(self):
       pass

    def __eq__(self, other):
        return np.equal(other.board,self.board).all() and other.player == self.player and other.winner == self.winner

    def __hash__(self):
        return hash((self.board.data.tobytes(),self.player,self.winner))


class Connect4MiniNode(Connect4Node):

    def __init__(self,env: Connect4,board: np.array, player: int, winner = 0):
        super().__init__(env,Connect4MiniNode,board,player, winner )

    def reward(self, player):
        reward = 1
        if self.winner == -1:
            return 0
        elif int(self.winner) != player:
            reward = -1
        return reward


class Connect4MCTSNode(Connect4Node):

    def __init__(self, env: Connect4, board: np.array, player: int, winner=0,root=False):
        super().__init__(env, Connect4MCTSNode,board, player, winner)

        self.root = root

    def reward(self) -> float:
        """
        if self.winner == -1:
            return 0.5
        elif int(self.winner) != self.player:
            reward = 1
            return reward
        """
        if self.winner == -1:
            return 0
        else:
            return 1


class Connect4MCTS(MCTS):

    def __init__(self,env: Connect4, n_sim: int = 50,c: int = 1,player = 1):
        super().__init__(c)

        print(f"Loaded MCTS with nsim = {n_sim} c = {c} player = {player}")
        self._env = env
        self._num_sim = n_sim
        self._player = player

    def act(self,board):

        s = Connect4MCTSNode(self._env,board,self._player,0)
        # First rum simulations to collect
        # Tree statistics
        for _ in range(self._num_sim):

            self.run(s)

        a = self.get_max_action(s)

        return a

    def get_max_action(self,s) -> int:

        if s not in self.children:
            raise Exception(f"Node {s} does not exist in the tree")

        if self._env.check_winner(s.board):
            raise Exception("Attempting to compute action on a leaf node")

        children = self.children

        values = [self.value(c) for c in children[s]]

        max_child = children[s][np.argmax(np.array(values))]

        action = self.get_parent_ation(s,max_child)

        return action

    def get_parent_ation(self,parent,child):

        a = parent.board

        b = child.board

        diff = a - b
        diffs = np.where(diff != 0)

        action = diffs[1][0]

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


class UCIOracle:
    """
    Agent trained from UCI Connect4 Data
    """
    @classmethod
    def load_from_disk(cls,env,nn_path,hidden_layer_size):

        inputDim = 42  # takes variable "x"
        outputDim = 1  # takes variable "y"

        model = UCINet(inputDim, hidden_layer_size, outputDim)

        model.load_state_dict(torch.load(nn_path))

        model.eval()

        return cls(env,nn=model)

    def __init__(self,env,data_path="",nn=None):

        self.env = env

        self._data = None
        self._data_path = data_path
        self._nn = nn
        self._pi = dict()

    def predict(self,x):
        return self._nn(x)

    def compute_policy(self):
        """
        Using the nn as a value function. Set the policy
        to a greedy policy
        :return:
        """
        if not self._data and not self._data_path:
            print("There is no data cant compute a policy")
            return None

        # TODO this assumes our value network is good
        # TODO we could also just use the labels

        X , Y = self.process_data(self._data_path)
        for i in range(len(X)):

            board = X[i]
            children = self.get_children(board)
            child_values = np.array([self._nn[c] for c in children])
            max_a = np.argmax(child_values)[0]
            self._pi[self.hash_numpy(board)] = max_a

    def convert_to_numpy(self,x):
        """
        Function to take the UCI format for connect 4
        and convert it to numpy.
        :param x:
        :return:
        """

        # TODO this assumes a fixed board dim
        new_x = np.zeros((6, 7))
        index = 0
        for col_i in range(7):
            col = []
            for row_i in range(5, -1, -1):
                new_x[row_i][col_i] = int(x[index])
                index += 1

        new_list = []
        for row_i in range(6):
            for col_i in range(7):
                new_list.append(new_x[row_i][col_i])

        return np.array(new_list)

    def get_children(self,board):

        valid_actions = self.env.valid_actions(board)
        curr_board = board.copy()
        children = []
        player = self.get_current_player(curr_board)

        for a in valid_actions:
            board_row = self.env.process_move(curr_board, a)
            next_board = curr_board.copy()
            next_board[board_row, a] = player
            children.append(next_board)
        return children

    def get_current_player(self,board):
        """
        Determine whos turn it is from the raw board inputs
        :param board:
        :return:
        """
        num_1 = len(np.where(board == 1)[0])
        num_2 = len(np.where(board == 2)[0])

        if num_2 < num_1:
            return 2
        else:
            return 1

    def hash_numpy(self,x):
        return hash(x.data.tobytes())

    def process_data(self,data_path):
        """
        Read in .data file and return structured
        data in numpy array
        :param data_path:
        :return:
        """
        data_file = open(data_path, "r")

        data = data_file.readlines()

        X_data, Y_data = [], []
        for i, d in enumerate(data):

            a = d.replace("b", "0").replace("x", "1").replace("o", "2").replace("\n", "").split(",")
            X = a[:-1]
            X = self.convert_to_numpy(X)
            Y = -1.0
            if "win" in a[-1]:
                Y = 1.0
            elif "draw" in a[-1]:
                Y = 0.0
            X_data.append(X)
            Y_data.append(Y)

        return np.array(X_data).astype(float), np.array(Y_data).reshape((len(Y_data), 1))

    def train_model(self):
        pass
