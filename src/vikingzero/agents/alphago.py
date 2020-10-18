import copy
import matplotlib.pyplot as plt
import neptune
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple,defaultdict
from dataclasses import dataclass
from operator import itemgetter
from torch.autograd import Variable

from ..search import MCTS
from ..designers.connect4_designer import Designer
from ..agents.tictactoe_agents import TicTacMCTSNode
from ..agents.connect4_agent import Connect4MCTSNode

Memory = namedtuple('Transition',
                        ('state', 'action', 'action_dist', 'value', 'z'))

@dataclass
class Memory:
    state: np.array
    action_dist: np.array
    action_mcts: np.array
    value: np.float
    z: np.float

class CnnSmall3(nn.Module):
    def __init__(self,width, height,  action_size, num_channels):
        # game params
        self.action_size = action_size
        self.board_x, self.board_y = width, height
        self.num_channels = num_channels

        super(CnnSmall3, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, 3)
        #self.pool = nn.MaxPool2d(2, 1)
        self.conv2 = nn.Conv2d(18, 16, 1)
        self.fc1 = nn.Linear(16 , 32)
        self.fc2 = nn.Linear(32, 9)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(-1,self.board_y, self.board_x,3)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x.view(-1,16)))
        pi = self.fc2(x)  # batch_size x action_size
        v = self.fc3(x)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # preparing input
        board = board.view(1, self.board_y, self.board_x,3)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        p,v =  torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

        return p,v[0]


class CnnSmall2(nn.Module):
    def __init__(self,width, height,  action_size, num_channels):
        # game params
        self.action_size = action_size
        self.board_x, self.board_y = width, height
        self.num_channels = num_channels

        super(CnnSmall2, self).__init__()
        self.conv1 = nn.Conv2d(1, self.num_channels, 3)

        self.fc1 = nn.Linear(self.num_channels, 9)

        self.fc2 = nn.Linear(9, self.action_size)

        self.fc3 = nn.Linear(9, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s[s == 2] = -1
        s = s.view(-1, 1, self.board_y, self.board_x)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.conv1(s))                       # batch_size x num_channels x board_x x board_y
        s = s.view(-1,self.num_channels)
        s = F.relu(self.fc1(s))
        pi = self.fc2(s)                                                                         # batch_size x action_size
        v = self.fc3(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # preparing input
        board = board.view(1, self.board_y, self.board_x)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        p,v =  torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

        return p,v[0]


class CnnNNetSmall(nn.Module):

    def __init__(self,width, height,  action_size, num_channels,dropout):
        # game params
        self.action_size = action_size
        self.board_x, self.board_y = width, height
        self.dropout = dropout
        self.num_channels = num_channels

        super(CnnNNetSmall, self).__init__()
        self.conv1 = nn.Conv2d(3, self.num_channels, 3,padding=2)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels,5,padding=2)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels,5)


        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        self.bn3 = nn.BatchNorm2d(self.num_channels)

        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc_bn1 = nn.BatchNorm1d(self.num_channels)

        self.fc2 = nn.Linear(self.num_channels, self.action_size)

        self.fc3 = nn.Linear(self.num_channels, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s[s == 2] = -1
        s = s.view(-1,3,self.board_y, self.board_x)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x board_x x board_y
        s = s.view(-1, self.num_channels)

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1024

        pi = self.fc2(s)                                                                         # batch_size x action_size
        v = self.fc3(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # preparing input
        #board = board.view(1, self.board_y, self.board_x,3)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        p,v =  torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

        return p,v[0]


class CnnNNet(nn.Module):
    def __init__(self,width, height,  action_size, num_channels,dropout):
        # game params
        self.action_size = action_size
        self.board_x, self.board_y = width, height
        self.dropout = dropout
        self.num_channels = num_channels

        super(CnnNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        self.bn3 = nn.BatchNorm2d(self.num_channels)
        self.bn4 = nn.BatchNorm2d(self.num_channels)

        self.fc1 = nn.Linear(self.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 3, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.num_channels*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # preparing input
        #board = torch.FloatTensor(board.astype(np.float64))
        #board = board.view(-1, self.board_x, self.board_y)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        p,v =  torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

        return p,v[0]


class Network(nn.Module):

    def __init__(self,input_size,output_size):
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size
        # This represents the shared layer(s) before the different heads
        # Here, I used a single linear layer for simplicity purposes
        # But any network configuration should work
        self.h1 = nn.Linear(input_size, input_size*2)
        self.h2 = nn.Linear(input_size*2, input_size*3)
        self.h3 = nn.Linear(input_size*3, input_size*4)
        self.h4 = nn.Linear(input_size*4, input_size*3)
        self.h5 = nn.Linear(input_size*3, input_size*2)
        self.h6 = nn.Linear(input_size*2, input_size)
        #self.h4 = nn.Linear(18, 9)

        # Set up the different heads
        # Each head can take any network configuration
        self.policy = nn.Linear(input_size , output_size)
        self.value = nn.Linear(input_size, 1)

    def forward(self, x):

        x[x == 2] = -1
        # Run the shared layer(s)
        x = x.view(-1, self._input_size)
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        x = F.relu(self.h4(x))
        x = F.relu(self.h5(x))
        x = F.relu(self.h6(x))

        # Run the different heads with the output of the shared layers as input
        p = F.log_softmax(self.policy(x), dim=1)
        value_out = torch.tanh(self.value(x))

        return p, value_out

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # preparing input
        #board = board.view(1, self.board_y, self.board_x)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        p, v = torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

        return p, v[0]


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
        data = random.sample(self.memory, batch_size)
        return [(d.state,d.action_mcts,d.z) for d in data]

    def save(self,path):

        for d in self.memory:
            print("---------BOARD-------------")
            board = d.state.reshape((3,3))
            print(board)
            print("Result value")
            print(f"Z = {d.z}")
            print("Action Probs")
            print(f"{d.action_mcts}")


    def __len__(self):
        return len(self.memory)


class AlphaZero(MCTS):

    def __init__(self,env, augment_input: bool = True, n_sim: int = 50, batch_size: int = 10,max_mem_size: int = 1000,
                 epochs: int = 10, c: int = 1, lr: float = 0.001, epsilon: float = 0.2,input_width: int = 3, input_height: int = 3,
                 output_size: int = 9,player: int = 1,momentum: float = 0.9, network_type: str = "normal",optimizer: str = "Adam", t_threshold: int = 10
                 ,test_name: str = "current",num_channels: int = 512, dropout: float = 0.3, weight_decay: float = 0.001,
                 eval_threshold: int = 1, dirichlet_noise: float = 0.3, network_path: str = ""):
        super().__init__(c)

        self.player = player

        self._action_size = output_size
        self._act_max = False
        self._augment_input = augment_input
        self._batch_size = batch_size
        self._current_memory = [] # hold memory for current game
        self._current_moves = 0 # Track action count
        self._dir_noise = dirichlet_noise
        self._dropout = dropout
        self._env = env
        self._epochs = epochs
        self._epsilon = epsilon
        self._eval_threshold = eval_threshold
        self._input_height = input_height
        self._input_width = input_width
        self._max_mem_size = max_mem_size
        self._memory = self.create_memory()
        self._network_path = network_path
        self._num_channels = num_channels
        self._nn = self.create_model(input_width,input_height,output_size,network_type)
        self._n_sim = n_sim
        if optimizer == "Adam":
            self._optimizer = torch.optim.Adam(self._nn.parameters(),lr=lr,weight_decay=weight_decay)
        elif optimizer == "SGD":
            self._optimizer = torch.optim.SGD(self._nn.parameters(),lr=lr,momentum=momentum)
        else:
            print("Passed incorrect optimizer. Using SGD by default")
            self._optimizer = torch.optim.SGD(self._nn.parameters(),lr=lr,momentum=momentum)

        self._softmax = torch.nn.Softmax(dim=1)
        self._tau = 1
        self._test_name = test_name
        self._t_threshold = t_threshold
        self._v_loss = torch.nn.MSELoss(reduction='sum')
        self._weight_decay = weight_decay

        if self._env.name == "TicTacToe":
            self._Node = TicTacMCTSNode
        elif self._env.name == "Connect4":
            self._Node = Connect4MCTSNode

    def act(self,board):

        s = self._Node(self._env,board,self._env.check_turn(board),0,root=True)
        # First rum simulations to collect
        # Tree statistics
        for _ in range(self._n_sim):

            self.run(s)

        a,p_a = self.get_action(s)

        board = self.processs_board(board)

        memory = Memory(board,None,p_a,None,None)

        self._current_memory.append(memory)

        self._current_moves += 1

        #assert self._current_moves <= 10
        #assert len(self._current_memory) <= 10

        return a, p_a

    def create_model(self,width,height,output_size,nn_type):

        if nn_type == "normal":
            input_size = width*height*3
            return Network(input_size,output_size)
        elif nn_type == "cnn":
            model =  CnnNNet(width,height,self._action_size,self._num_channels,self._dropout)
        elif nn_type == "cnn_small":
            model =  CnnNNetSmall(width, height, self._action_size, self._num_channels, self._dropout)
        elif nn_type == "cnn_small2":
            model =  CnnSmall2(width, height, self._action_size, self._num_channels)
        elif nn_type == "cnn_small3":
            model =  CnnSmall3(width, height, self._action_size, self._num_channels)
        else:
            raise Exception(f"No neural network type available for {nn_type}")

        if len(self._network_path) > 1:
            print("Loading model from disk")
            model.load_state_dict(torch.load(self._network_path))

        return model

    def create_memory(self):
        return ReplayMemory(self._max_mem_size)

    def eval(self):
        self._act_max = True

    def get_action(self,s) -> tuple:
        """
        Sample action based on statistics from tree
        :param s:
        :return:
        """

        #TODO Optim
        temp_power = 1.0 / self._tau

        children = self.children[s]

        actions = self.get_valid_actions(s)

        c_counts = np.array([self._N[c]**temp_power for c in children])

        c_sum = sum(c_counts)

        p_a = c_counts / c_sum

        p = np.zeros(self._action_size)

        p[actions] = p_a

        child_act = np.random.choice(children, p=p_a)

        s_p , s_v = self.predict(s)

        if self._act_max or (self._current_moves > self._t_threshold):
            #TODO clean up this mess
            # view network
            net_view = list(zip(*[self.predict(c) for c in children]))

            p_view = np.array(net_view[0])
            v_view = net_view[1]

            q_values , counts = list(zip(*[(self._Q[c],self._N[c]) for c in children]))
            q_values = np.array(q_values)
            c = np.zeros(self._action_size)
            v = c.copy()
            q = c.copy()
            ucb = c.copy()
            w = c.copy()
            p_net = c.copy()
            c[actions] = counts
            v[actions] = v_view
            q[actions] = q_values

            if self._act_max:
                print(f" Value of current node = {s_v}")

                if self._env.name == "TicTacToe":

                    print("----------- COUNT ----------------")
                    print(c.reshape((self._input_height,self._input_width)))

                    print("----------- Values ----------------")
                    print(v.reshape((self._input_height,self._input_width)))

                    print("----------- MCTS Policy ----------------")
                    print(p.reshape((self._input_height, self._input_width)))

                    print("----------- NN Policy ----------------")
                    print(s_p.reshape((self._input_height, self._input_width)))

                    print("----------- Q values ---------------")
                    print(q.reshape((self._input_height,self._input_width)))
                else:

                    print("----------- COUNT ----------------")
                    print(c)

                    print("----------- Values ----------------")
                    print(v)

                    print("----------- MCTS Policy ----------------")
                    print(p)

                    print("----------- NN Policy ----------------")
                    print(s_p)

                    print("----------- Q values ---------------")
                    print(q)

            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestAs = np.random.choice(bestAs)
            child_act = children[bestAs]
            parent_act = self.get_parent_action(s,child_act)

            def uct(child,Q,N,p):
                a = self.get_parent_action(s,child)
                value_term = Q[child] / (N[child] + 1)
                ucb_term = self.c*p[a]*(np.sqrt(N[s]) / (1 + N[child]))
                return value_term,ucb_term

            w_term, ucb_term = list(zip(*[uct(child,self._Q,self._N,p) for child in children]))
            w[actions] = np.array(w_term)
            ucb[actions] = np.array(ucb_term)
            if self._act_max:

                if self._env.name == "TicTacToe":
                    print("-------------------- W Values -----------------------------")
                    print(w.reshape((self._input_height, self._input_width)))

                    print("-------------------- UCB Values -----------------------------")
                    print(ucb.reshape((self._input_height, self._input_width)))
                else:
                    print("-------------------- W Values -----------------------------")
                    print(w)

                    print("-------------------- UCB Values -----------------------------")
                    print(ucb)

            p = np.zeros(self._action_size)
            p[parent_act] = 1
            return parent_act, p

        return self.get_parent_action(s,child_act), p

    def get_parent_action(self,parent,child):

        a = parent.board
        b = child.board

        diff = a - b
        diffs = np.where(diff != 0)

        if a.shape[0] == 9:  # tictactoe
            action = diffs[0][0]
        elif len(a.flatten()) == 42:  # Connect4
            action = diffs[1][0]

        return action

    def get_valid_actions(self,s):
        """
        Take in node and return the valid actions from that node
        :param n:
        :return:
        """
        board = s.board

        return self._env.valid_actions(board)

    def node_to_board(self,s):

        board = s.board
        board = self.processs_board(board)
        board = Variable(torch.from_numpy(board).type(dtype=torch.float))

        return board

    def predict(self,s):
        board = self.node_to_board(s)
        return self._nn.predict(board)

    def processs_board(self,board):

        board = board.reshape((self._input_height, self._input_width))

        if self._augment_input:
            board = self.transform_board(board)

        return board

    def reset(self):
        """
        Reset the agent before playing a new game
        :return:
        """
        self._current_moves = 0
        self.reset_current_memory()
        self.reset_tree()

    def reset_current_memory(self):
        self._current_memory = []

    def reset_memory(self):
        self._memory.memory = []
        self._memory.position = 0

    def reset_tree(self):
        self._Q = defaultdict(int)
        self._N = defaultdict(int)

        self.children = dict()

    def reverse_transform(self,board):
        """
        Take board state from transformed state back to original state
        :param board:
        :return:
        """

        board_original = np.zeros((self._input_height*self._input_width,))

        # first board in transformed board is player 1
        p1_actions = np.where(board[0].flatten() == 1)[0]
        #if len(p1_actions) > 0:
        #    p1_actions = p1_actions[0]
        board_original[p1_actions] = 1
        p2_actions = np.where(board[1].flatten() == 1)[0]
        #if len(p2_actions) > 0:
        #    p2_actions = p2_actions[0]

        board_original[p2_actions] = 2

        return board_original

    def reward(self,leaf_node):

        winner = leaf_node.winner
        if winner == -1:
            return 0

        if winner == leaf_node.player:
            return 1
        else:
            return -1

    def save(self,id):
        torch.save(self._nn.state_dict(), f"current_best_{self._env.name}_{id}")

    def select_child(self,node):

        children = self.children[node]

        actions = self.get_valid_actions(node)

        p , v = self.predict(node)

        #p = p.detach().numpy().reshape(self._action_size)
        # renormalize
        p[[a for a in range(self._action_size) if a not in actions]] = 0

        p = p / p.sum()

        N_p = self._N[node]

        if node.root:
            dir_noise = np.zeros(self._action_size)

            dir_noise[actions] = np.random.dirichlet([self._dir_noise]*len(actions))

            p = (1 - self._epsilon)*p + self._epsilon*dir_noise

        child_v = []
        for child in children:
            a = self.get_parent_action(node, child)
            p_a = p[a]
            u_c = self._Q[child] / (self._N[child] + 1) + self.c * p_a * (np.sqrt(N_p) / (1 + self._N[child]))
            child_v.append((child, u_c))

        # return max(self.children[node], key=uct)

        return max(child_v, key=itemgetter(1))[0]

    def simulate(self,node):

        if node.is_terminal():
            #reward = node.reward()
            reward = self.reward(node)
            return (node.player, reward)

        p,v = self.predict(node)
        #winner = 1 if node.player == 2 else 2

        return (node.player,float(v))

    def show_data_sample(self,sample):
        """
        Method to help get a visual on batch of training data
        :param sample:
        :return:
        """

        for data in sample:
            b , p , v = data

            board = self.reverse_transform(b)
            print("-----------------BOARD---------------------")
            print(board.reshape((3,3)))
            print("MCTS PROBS")
            print(p.reshape((3,3)))
            print(f"Value = {v}")

    def store_memory(self,z):
        """
        Take result from game and
        :param z:
        :return:
        """
        #z = torch.IntTensor(z)
        for mem in self._current_memory:

            board = mem.state
            if self._augment_input:
                board = self.reverse_transform(mem.state)
            p_turn = self._env.check_turn(board)

            if z == -1:
                mem.z = 0.000001
                #mem.z = 0
            elif p_turn == z: # if winner != player of node
                mem.z = 1
            elif p_turn != z:
                mem.z = -1
            else:
                raise Exception("Incorrect winner passed")

            """
            if z == 2:
                print("player 2 won")
            print("--------- MEM STATE ----------------")
            #print(mem.state.reshape((self._input_height, self._input_width)))
            print(mem.state)
            print("--------- BOARD STATE ----------------")
            print(board.reshape((3,3)))
            print(f"Turn = {p_turn}")
            print(f"z = {z}")
            print(f"mem.z = {mem.z}")
            """
            self._memory.push(mem)

        self.reset_current_memory()

    def view_current_memory(self):

        last_n = 20
        data = self._memory.memory[-last_n:]

        for d in data:
            b,p,v = d.state,d.action_mcts,d.z
            board = self.reverse_transform(b)
            print("-----------------BOARD---------------------")
            print(board.reshape((3, 3)))
            print("MCTS PROBS")
            print(p.reshape((3, 3)))
            print(f"Value = {v}")

    def train_network(self):

        self._nn.train()

        if len(self._memory) < self._batch_size:
            return None,None,None

        avg_total = 0
        avg_value = 0
        avg_policy = 0

        #self.view_current_memory()

        for _ in range(self._epochs):

            data = self._memory.sample(self._batch_size)

            b, p, v = zip(*[d for d in data])

            target_p = torch.FloatTensor(p)
            target_v = torch.FloatTensor(v)
            state = torch.FloatTensor(b)

            p, v = self._nn(state)

            loss_value = self.loss_v(target_v,v)

            loss_policy = self.loss_pi(target_p, p)

            loss = loss_value + loss_policy #loss_policy

            avg_total += loss
            avg_value += loss_value
            avg_policy += loss_policy

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()


        return avg_total / self._epochs, avg_policy / self._epochs, avg_value / self._epochs

    def train(self):
        self._act_max = False

    def transform_board(self,board):
        #board = board.reshape((self._input_height,self._input_width))
        p1 = np.zeros(board.shape)
        p2 = np.zeros(board.shape)
        p = np.ones(board.shape)

        p1[board == 1] = 1
        p2[board == 2] = 1
        turn = self._env.check_turn(board)
        p = p if turn == 1 else p*-1
        board = np.stack((p1,p2,p))
        return board

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):

        return torch.sum((targets - outputs) ** 2) / targets.size()[0]


class DesignerZero(Designer):

    def __init__(self,env,agent_config,exp_config,_run=False, eval_threshold = 1):
        super().__init__(env,agent_config,exp_config, _run=_run)

        self.eval_threshold = eval_threshold
        self._train_iters = exp_config["train_iters"]
        self._run_evaluator = exp_config["run_evaluator"]
        self.current_best = self.load_agent(self._agent1_config)
        self.current_player = copy.deepcopy(self.current_best)
        self.exp_id = self.load_exp_id()

        if not self._run:
            try:
                self._run= neptune.get_experiment()
            except:
                pass

        ###########
        # LOSS
        ###########
        self.avg_loss = []
        self.avg_value_loss = []
        self.avg_policy_loss = []

    def load_exp_id(self):
        if self._run:
            return self._run.id
        else:
            try:
                return neptune.get_experiment().id
            except:
                pass

        return None

    def play_game(self,render,agent1,agent2,iter=None,game_num=0):

        self.env.reset()

        try:
            agent1.reset()
        except:
            pass
        try:
            agent2.reset()
        except:
            pass

        curr_player = agent1

        game_array = []

        while True:

            action = curr_player.act(self.env.board)
            if type(action) == tuple:
                action , p_a = action

            if self._record_all:
                curr_board = self.env.board.copy()
                b_hash = hash((curr_board.tobytes(),))
                #self._run.info[f"action_iter={iter}_{b_hash}_{game_num}"] = (curr_board.tolist(),int(action))

            curr_state, action, next_state, r = self.env.step(action)

            if render:
                game_array.append(self.env.board.copy().tolist())
                self.env.render()

            if r != 0:
                break

            curr_player = agent2 if curr_player == agent1 else agent1

        return self.env.winner

    def run(self):
        """
            0: Self Play - Store memories
            1: Train network
            2: Evaluate - Decide current best player
            :return:
        """

        # Initialize run by training a current best agent
        # self.train(self.current_best,self._train_iters)

        vs_minimax = []
        vs_best = []
        for iter in range(self._iters):

            print(f"Running iteration {iter}")

            # Self Play
            s_time = time.time()
            self.train(self.current_player,self._train_iters)
            e_time = time.time()
            min_time = (e_time - s_time) / 60.0
            print(f"Training time ={min_time} For {self._train_iters} iters")

            # Train Network
            s_time = time.time()
            avg_total, avg_policy, avg_val = self.current_player.train_network()

            if self._run:
                neptune.log_metric("Total Loss", avg_total)
                neptune.log_metric("Value loss", avg_val)
                neptune.log_metric("Policy loss", avg_policy)
            e_time = time.time()
            min_time = (e_time - s_time) / 60.0
            print(f"Training network time ={min_time}")

            if (iter % self._record_every) == 0:
                # Evaluate
                print(" ---------- Eval as player 1 vs minimax ---------")
                p1_result = self.run_eval(self.current_player, self.agent2,self._eval_iters,iter=iter)
                vs_minimax.append(p1_result)
                if self._run:
                    neptune.log_metric("tot_p1_wins", p1_result)

                print(" ---------- Eval as player 2 vs minimax ---------")
                p2_result = self.run_eval(self.agent2, self.current_player, self._eval_iters, iter=iter)
                p2_result *= -1
                vs_minimax.append(p2_result)
                if self._run:
                    neptune.log_metric("tot_p2_wins", p2_result)


            if self._run_evaluator:
                print("---------- Current Player vs Current Best ____________ ")
                curr_result = self.run_eval(self.current_player,self.current_best,10,iter=iter)

                curr_result2 = self.run_eval(self.current_best,self.current_player,10,iter=iter)

                tot_result = curr_result + -1*curr_result2

                vs_best.append(tot_result)

                if (tot_result >= self.eval_threshold):
                    print(f"Changing Agent on iteration = {iter}")
                    self.current_best = copy.deepcopy(self.current_player)
                else:
                    self.current_player = copy.deepcopy(self.current_best)

                if self._run:
                    neptune.log_metric("currp_vs_currbest",tot_result)

            if self._run:
                self.current_player.save(self.exp_id)
        """
        plt.plot(self.avg_loss,label="Avg total loss")
        plt.legend()
        plt.show()
        plt.plot(self.avg_value_loss,label="Avg val loss")
        plt.legend()
        plt.show()
        plt.plot(self.avg_policy_loss,label="Avg policy loss")
        plt.legend()
        plt.show()
        """

    def run_eval(self,agent1,agent2,iters,render=False,iter=None):

        """
            This method evaluates the current agent
            :return:
        """

        agent1.player = 1
        agent2.player = 2

        try:
            agent1.eval()
        except:
            pass
        try:
            agent2.eval()
        except:
            pass
        result = 0
        for i in range(iters):
            winner = self.play_game(self._render, agent1, agent2, iter=iter,game_num=i)
            if winner == agent1.player:
                result += 1
            elif winner == agent2.player:
                result -= 1

        return result

    def train(self,agent,iters):

        agent.reset_current_memory()

        agent.train()

        for _ in range(iters):

            z = self.play_game(False,agent,agent)

            agent.store_memory(z)


