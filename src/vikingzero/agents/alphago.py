import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.policy = nn.Linear(9 , 9)
        self.value = nn.Linear(9, 1)

    def forward(self, x):

        # Run the shared layer(s)
        x = self.shared_layer(x)

        # Run the different heads with the output of the shared layers as input
        policy_out = F.log_softmax(self.policy(x), dim=1)
        value_out = self.value(x)

        return policy_out, value_out


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, mem):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = mem
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class AlphaZero(MCTS):

    def __init__(self,env,node, n_sim: int = 50, batch_size=10, c: int = 1,
                 player: int = 1):
        super().__init__(c)

        self._action_size = 9
        self._batch_size = batch_size
        self._current_memory = [] # hold memory for current game
        self._env = env
        self._memory = self.create_memory()
        self._nn = self.create_model()
        self._n_sim = n_sim
        self._optimizer = torch.optim.Adam(self._nn.parameters())
        self._player = player
        self._softmax = torch.nn.Softmax(dim=1)
        self._tau = 1

        self._v_loss = torch.nn.MSELoss(reduction='sum')

        self._Node = node

    def act(self,board,render=False):

        s = self._Node(self._env,board,self._env.current_player,0)
        # First rum simulations to collect
        # Tree statistics
        for _ in range(self._n_sim):

            self.run(s)

        a,p_a = self.get_action(s)

        if len(board.shape) == 1:
            board = board.reshape((1, board.shape[0]))

        board = Variable(torch.from_numpy(board).type(dtype=torch.float))

        #p, v = self._nn(board)

        memory = Memory(board,None,p_a,None,None)

        self._current_memory.append(memory)

        return a

    def get_action(self,s) -> tuple:
        """
        Sample action based on statistics from tree
        :param s:
        :return:
        """

        #TODO Optim
        temp_power = 1.0 / self._tau

        children = self.children[s]

        actions = [self.get_parent_action(s,c) for c in children]

        c_counts = np.array([self._N[c]**temp_power for c in children])

        c_sum = sum(c_counts)

        p_a = c_counts / c_sum

        child_act =np.random.choice(children,p=p_a)

        p = np.zeros(self._action_size)

        p[actions] = p_a

        return self.get_parent_action(s,child_act), p

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

        p = p.reshape(self._action_size)

        N_p = self._N[node]

        child_v = []
        for child in children:
            a = self.get_parent_action(node, child)
            p_a = p[a]
            u_c = self._Q[child] / self._N[child]+ self.c * p_a * (np.sqrt(N_p) / (1 + self._N[child]))
            child_v.append((child, u_c))

        # return max(self.children[node], key=uct)

        return max(child_v, key=itemgetter(1))[0]

    def simulate(self,node):

        board = node.board
        if len(board.shape) == 1:
            board = board.reshape((1,board.shape[0]))
        board = Variable(torch.from_numpy(board).type(dtype=torch.float))

        p,v = self._nn(board)

        return (node.winner,float(v))

    def store_memory(self,z):
        """
        Take result from game and
        :param z:
        :return:
        """
        #z = torch.IntTensor(z)
        for mem in self._current_memory:

            p_turn = self._env.check_turn(mem.state)

            if p_turn != z: # if winner != player of node
                mem.z = 1
            else:
                mem.z = 0

            self._memory.push(mem)

    def train_network(self):

        if len(self._memory) < self._batch_size:
            return

        data = self._memory.sample(self._batch_size)

        for d in data:


            target = torch.FloatTensor(d.action_mcts.reshape(self._action_size))

            p, v = self._nn(d.state)
            loss_value = (d.z - v)** 2
            loss_policy = self.loss_pi(target, p)

            loss = loss_value + loss_policy #loss_policy
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def create_model(self):
        return Network()

    def create_memory(self):
        return ReplayMemory(10000)

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):

        #return torch.sum((targets - outputs) ** 2)
        #return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
        pass


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

            self.current_player.store_memory(z)
