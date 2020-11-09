import copy
import matplotlib.pyplot as plt
try:
    import neptune
except:
    pass
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple
from dataclasses import dataclass
from torch.autograd import Variable

from ..search import ZeroMCTS, ZeroNode

Memory = namedtuple('Transition',
                        ('state', 'action', 'action_dist', 'value', 'z'))

@dataclass
class Memory:
    state: np.array
    action_dist: np.array
    action_mcts: np.array
    value: np.float
    z: np.float


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
            state = d.state.reshape((3,3))
            print(state)
            print("Result value")
            print(f"Z = {d.z}")
            print("Action Probs")
            print(f"{d.action_mcts}")


    def __len__(self):
        return len(self.memory)


class AlphaZero:

    def __init__(self,env, augment_input: bool = True, n_sim: int = 50, batch_size: int = 10,max_mem_size: int = 1000,
                 epochs: int = 10, c: int = 1, lr: float = 0.001, epsilon: float = 0.2,input_width: int = 3, input_height: int = 3,
                 output_size: int = 9,player: int = 1,momentum: float = 0.9, network_type: str = "normal",optimizer: str = "Adam", t_threshold: int = 10
                 ,test_name: str = "current",num_channels: int = 512, dropout: float = 0.3, weight_decay: float = 0.001,
                 eval_threshold: int = 1, dirichlet_noise: float = 0.3, network_path: str = ""):

        self.current_state = None # Used for tracking state. Used for tree lookup
        self.prev_state = None
        self.player = player

        self._action_size = output_size
        self._act_max = False
        self._augment_input = augment_input
        self._batch_size = batch_size
        self._c = c
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

        self.MCTS = ZeroMCTS(self._env,self._nn,self.node_to_state,
                             self._c,dir_noise=self._dir_noise,dir_eps=self._epsilon)

    def act(self,state):

        self.current_state = state.copy()

        s = self.get_node(self.current_state)

        self.MCTS.root = s
        # First run simulations to collect
        # Tree statistics
        for _ in range(self._n_sim):

            self.MCTS.run(s)

        a,p_a = self.get_action(s)

        state = self.processs_state(state)

        assert p_a.sum() > 0.9999

        memory = Memory(state,None,p_a,None,None)

        self._current_memory.append(memory)

        self._current_moves += 1

        self.prev_state = self.current_state.copy()

        return a, p_a

    def create_model(self,width,height,output_size,nn_type):

        if nn_type == "normal":
            input_size = width*height*3
            return Network(input_size,output_size)
        elif nn_type == "cnn":
            model =  CnnNNet(width,height,self._action_size,self._num_channels,self._dropout)
        elif nn_type == "cnn_small":
            model =  CnnNNetSmall(width, height, self._action_size, self._num_channels, self._dropout)
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
        self.MCTS.act_max = True

    def get_action(self,s) -> tuple:
        """
        Sample action based on statistics from tree
        :param s:
        :return:
        """

        tree = self.MCTS

        if self._act_max or (self._current_moves > self._t_threshold):
            a, p = tree.policy(s, self._tau, max=True)

        else:
            a, p = tree.policy(s, self._tau)

        return a , p

    def get_node(self,state):
        """
        Create a Node from the given state
        @param state: np.array
        @return: Node Object
        """
        parent_action = None
        if self.prev_state is not None:
            parent_action = self.get_parent_action(self.prev_state,self.current_state)
        player = self._env.check_turn(state)
        winner = self._env.check_winner(state)

        root = ZeroNode(state=state, player=player, winner=winner, parent=None, parent_action=parent_action)


        if len(self.MCTS.dec_pts) == 0:
            self.MCTS.children.append([])
            self.MCTS.dec_pts.append(root)
            self.MCTS.parents.append(None)

        return root

    def get_parent_action(self,parent,child):

        if type(parent) == np.array or type(parent) == np.ndarray:
            a = parent
            b = child
        else:
            a = parent.state
            b = child.state

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
        state = s.state

        return self._env.valid_actions(state)

    def node_to_state(self,s):

        state = s.state
        state = self.processs_state(state)
        state = Variable(torch.from_numpy(state).type(dtype=torch.float))

        return state

    def predict(self,s):
        state = self.node_to_state(s)
        return self._nn.predict(state)

    def processs_state(self,state):

        state = state.reshape((self._input_height, self._input_width))

        if self._augment_input:
            state = self.transform_state(state)

        return state

    def reset(self):
        """
        Reset the agent before playing a new game
        :return:
        """

        self._current_moves = 0
        self.reset_current_memory()
        self._env.reset()
        self.current_state = None
        self.prev_state = None
        self.MCTS = ZeroMCTS(self._env,self._nn,self.node_to_state,
                             self._c,dir_noise=self._dir_noise,dir_eps=self._epsilon)
        self.MCTS.reset_tree()

    def reset_current_memory(self):
        self._current_memory = []

    def reset_memory(self):
        self._memory.memory = []
        self._memory.position = 0

    def reverse_transform(self,state):
        """
        Take state state from transformed state back to original state
        :param state:
        :return:
        """

        state_original = np.zeros((self._input_height*self._input_width,))

        # first state in transformed state is player 1
        p1_actions = np.where(state[0].flatten() == 1)[0]
        #if len(p1_actions) > 0:
        #    p1_actions = p1_actions[0]
        state_original[p1_actions] = 1
        p2_actions = np.where(state[1].flatten() == 1)[0]
        #if len(p2_actions) > 0:
        #    p2_actions = p2_actions[0]

        state_original[p2_actions] = 2

        return state_original

    def save(self,id):
        torch.save(self._nn.state_dict(), f"current_best_{self._env.name}_{id}")

    def show_data_sample(self,sample):
        """
        Method to help get a visual on batch of training data
        :param sample:
        :return:
        """

        for data in sample:
            b , p , v = data

            state = self.reverse_transform(b)
            print("-----------------state---------------------")
            print(state.reshape((3,3)))
            print("MCTS PROBS")
            print(p.reshape((3,3)))
            print(f"Value = {v}")

    def store_memory(self,z):
        """
        Take result from game and
        :param z:
        :return:
        """
        for mem in self._current_memory:

            state = mem.state
            if self._augment_input:
                state = self.reverse_transform(mem.state)
            p_turn = self._env.check_turn(state)

            if z == -1:
                mem.z = 0.0001
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
            print("--------- state STATE ----------------")
            print(state.reshape((3,3)))
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
            state = self.reverse_transform(b)
            print("-----------------state---------------------")
            print(state.reshape((3, 3)))
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
        self.MCTS.act_max = False

    def transform_state(self,state):
        state = state.reshape((self._input_height,self._input_width))
        p1 = np.zeros(state.shape)
        p2 = np.zeros(state.shape)
        p = np.ones(state.shape)

        p1[state == 1] = 1
        p2[state == 2] = 1
        turn = self._env.check_turn(state)
        p = p if turn == 1 else p*-1
        state = np.stack((p1,p2,p))
        return state

    def update_state(self,prev_state,curr_state):
        self.prev_state = prev_state
        self.current_state = curr_state

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):

        return torch.sum((targets - outputs) ** 2) / targets.size()[0]


