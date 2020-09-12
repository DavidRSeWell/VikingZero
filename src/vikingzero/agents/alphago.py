import copy
import numpy as np
import random
import torch
import torch.nn as nn

from collections import namedtuple
from dataclasses import dataclass
from operator import itemgetter
from torch.autograd import Variable

from ..search import MCTS
from ..designers.connect4_designer import Designer

Memory = namedtuple('Transition',
                        ('state', 'action', 'action_dist', 'value', 'z'))

@dataclass
class Memory:
    state: np.array
    action_dist: np.array
    action_mcts: np.array
    value: np.float
    z: np.float


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # This represents the shared layer(s) before the different heads
        # Here, I used a single linear layer for simplicity purposes
        # But any network configuration should work
        self.shared_layer = nn.Linear(9, 9)

        # Set up the different heads
        # Each head can take any network configuration
        self.policy = nn.Softmax(dim=1)
        self.value = nn.Linear(9, 1)

    def forward(self, x):

        # Run the shared layer(s)
        x = self.shared_layer(x)

        # Run the different heads with the output of the shared layers as input
        policy_out = self.policy(x)
        value_out = self.value(x)

        return policy_out, float(value_out)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Memory(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class AlphaZero(MCTS):

    def __init__(self,env,node, n_sim: int = 50, batch_size=100, c: int = 1,
                 player: int = 1):
        super().__init__(c)

        self._batch_size = batch_size
        self._current_memory = [] # hold memory for current game
        self._env = env
        self._memory = self.create_memory()
        self._nn = self.create_model()
        self._n_sim = n_sim
        self._player = player
        self._softmax = torch.nn.Softmax(dim=1)
        self._tau = 1

        self._Node = node

    def act(self,board,render=False):

        s = self._Node(self._env,board,self._env.current_player,0)
        # First rum simulations to collect
        # Tree statistics
        for _ in range(self._n_sim):

            self.run(s)

        a,p_a = self.get_action(s)

        board = Variable(torch.from_numpy(s.board).type(dtype=torch.float))

        p, v = self._nn(board)

        memory = Memory(board,p_a,p,v,None)

        self._current_memory.append(memory)

        return a

    def get_action(self,s) -> tuple:
        """
        Sample action based on statistics from tree
        :param s:
        :return:
        """

        temp_power = 1.0 / self._tau

        children = self.children[s]

        c_counts = [self._N[c]**temp_power for c in children]

        c_sum = sum(c_counts)

        p_a = c_counts / c_sum

        return np.random.choice(children,p_a) , p_a

    def get_parent_action(self,parent,child):

        a = parent.board

        b = child.board

        diff = a - b
        diffs = np.where(diff != 0)

        action = diffs[0][0]

        return action

    def select_child(self,node):

        children = self.children[node]

        board = node.board

        if len(board.shape) == 1:
            board = board.reshape((1, board.shape[0]))

        board = Variable(torch.from_numpy(board).type(dtype=torch.float))

        p , v = self._nn(board)

        N_p = self._N[node]

        child_v = []
        for child in children:
            a = self.get_parent_action(node, child)
            p_a = p[a]
            u_c = self._Q[child] + self.c * p_a * (np.sqrt(N_p) / (1 + self._N[child]))
            child_v.append((child, u_c))

        # return max(self.children[node], key=uct)

        return max(child_v, key=itemgetter(1))[0]

    def simulate(self,node):

        board = node.board
        if len(board.shape) == 1:
            board = board.reshape((1,board.shape[0]))
        board = Variable(torch.from_numpy(board).type(dtype=torch.float))

        p,v = self._nn(board)

        return v

    def store_memory(self,z):
        """
        Take result from game and
        :param z:
        :return:
        """
        for mem in self._current_memory:

            p_turn = self._env.check_turn(mem.board)

            if p_turn != z: # if winner != player of node
                mem.z = 1
            else:
                mem.z = 0

            self._memory.push(mem)

    def train_network(self):

        if len(self._memory) < self._batch_size:
            return

        data = self._memory.sample(self._batch_size)

    def create_model(self):
        return Network()

    def create_memory(self):
        return ReplayMemory(10^9)


class DesignerZero(Designer):

    def __init__(self,env,agent_config,exp_config,_run=False):
        super().__init__(env,agent_config,exp_config, _run=False)

        self._train_iters = exp_config["train_iters"]
        self.current_best = self.load_agent(self._agent1_config)
        self.current_player = copy.deepcopy(self.current_best)

    def run(self):
        """
            0: Self Play - Store memories
            1: Train network
            2: Evaluate - Decide current best player
            :return:
        """

        for iter in range(self._iters):

            print(f"Running iteration {iter}")

            if (iter % self._record_every) == 0 or (iter == self._iters - 1):

                r = self.run_eval(iter=iter)

                if self._run:
                    self._run.log_scalar(r"tot_wins",r)

            # Self Play
            self.train(self._train_iters)

            # Train Network
            self.current_player.train_network()

            # Evaluate

    def run_eval(self,iter=None):
        return 0

    def train(self,iters):

        for _ in range(iters):

            z = self.play_game(self._render,self.current_player,self.current_player)

            self.current_player.update_memory(z)
