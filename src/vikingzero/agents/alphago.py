import copy
import matplotlib.pyplot as plt
try:
    import neptune
except:
    pass
import numpy as np
import random
import time
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

    def __len__(self):
        return len(self.memory)


class AlphaZero:

    def __init__(self,env, augment_input: bool = True, n_sim: int = 50, batch_size: int = 10,max_mem_size: int = 1000,
                 epochs: int = 10, c: int = 1, lr: float = 0.001, epsilon: float = 0.2,input_width: int = 3, input_height: int = 3,
                 output_size: int = 9,player: int = 1,momentum: float = 0.9, network_type: str = "cnn_small",optimizer: str = "Adam", t_threshold: int = 10
                 ,test_name: str = "current",num_channels: int = 512, dropout: float = 0.3, weight_decay: float = 0.001,
                 eval_threshold: int = 1, dirichlet_noise: float = 0.3, network_path: str = "", mcts_policy_opt: bool = False, minimax_lookup_path="",
                 state_lookup_path=""):

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
        self._lr = lr
        self._max_mem_size = max_mem_size
        self._mcts_policy_opt = mcts_policy_opt
        self._memory = self.create_memory()
        self._momentum= momentum
        self._minimax_actions = self.load_lookup(minimax_lookup_path)
        self._network_path = network_path
        self._num_channels = num_channels
        self._nn = self.create_model(input_width,input_height,network_type)
        self._n_sim = n_sim
        self._optimizer = optimizer
        self._softmax = torch.nn.Softmax(dim=1)
        self._state_lookup = self.load_lookup(state_lookup_path)
        self._tau = 1
        self._test_name = test_name
        self._t_threshold = t_threshold
        self._v_loss = torch.nn.MSELoss(reduction='sum')
        self._weight_decay = weight_decay
        self.MCTS = ZeroMCTS(self._env,self._nn,self.node_to_state,
                             self._c,dir_noise=self._dir_noise,dir_eps=self._epsilon)

    def act(self,state):

        s = self.run_search(state)

        if self._mcts_policy_opt:
            s_time = time.time()
            a ,p_a = self.compute_pi_bar(s)
            e_time = time.time()
            rt = (e_time - s_time) / 60.0
            #print(f"MCTS POLICY OPT RT = {rt}")

        else:
            s_time = time.time()
            a, p_a = self.get_action(s)
            e_time = time.time()
            rt = (e_time - s_time) / 60.0
            #print(f"Rt norm = {rt}")

        state = self.process_state(state)

        assert p_a.sum() > 0.9999

        memory = Memory(state,None,p_a,None,None)

        self._current_memory.append(memory)

        self._current_moves += 1

        self.prev_state = self.current_state.copy()

        return a, p_a

    def compute_pi_bar(self, s) -> tuple:
        """
        Compute pi bar as described in the paper
        MCTS as regularized policy optimization
        https://arxiv.org/abs/2007.12509
        """

        def compute_lambda_multiplier(mcts, node, env) -> float:

            valid_actions = env.valid_actions(node.state)

            n = 0
            for a in valid_actions:
                n += mcts._Nsa[(node, a)]

            return self._c*np.sqrt(n) / (n + len(valid_actions))

        def compute_pi_alpha(p_nn, L, q, alpha):

            q = np.array(q)
            if q.sum() == 0:
                print("No q values!")
            p_nn = np.array(p_nn)
            denom = alpha - q
            denom[denom == 0] = -0.001
            assert 0 not in denom
            return L * p_nn / denom

        def find_max_min_alpha(p_a, L, q):

            a = q + L * p_a

            a = a[a != 0]

            min_alpha = np.max(a)

            b = q + L

            max_alpha = np.max(b)

            if min_alpha == 0:
                #min_alpha += 0.0000001
                pass

            return min_alpha, max_alpha

        def get_qa(mcts, node, p_nn):

            valid_actions = self._env.valid_actions(node.state)

            q = np.zeros((self._env.action_size,))

            qa = [mcts._Qsa[(node, a)] for a in valid_actions]

            q[valid_actions] = qa

            q = (q - np.min(q)) / (np.max(q) - np.min(q))

            p_nn[[a for a in range(self._env.action_size) if a not in valid_actions]] = 0

            p_nn /= p_nn.sum()

            return q, p_nn

        def find_max_alpha(node, env, mcts, p_nn) -> tuple:

            l_n = compute_lambda_multiplier(mcts, node, env)

            q, p_nn = get_qa(mcts, node, p_nn)

            min_a, max_a = find_max_min_alpha(p_nn, l_n, q)


            alpha_star = min_a

            delta = 10000
            pi_bar = None
            half = None

            epsilon = 0.01

            while True:

                if min_a == max_a:
                    return half,pi_bar
                half_distance = (min_a - max_a) / 2.0
                half = min_a - half_distance

                pi_bar = compute_pi_alpha(p_nn, l_n, q, half)

                if pi_bar.sum() > (1 + epsilon):

                    min_a = half

                elif pi_bar.sum() < (1 - epsilon):

                    max_a = half

                else:
                    #print("Found max alpha! ")
                    #print(half)
                    break

            return half, pi_bar

        p_nn, v = self.predict(s)

        valid_actions = self._env.valid_actions(s.state)

        mask = [a for a in range(p_nn.shape[0]) if a not in valid_actions]

        p_nn[mask] = 0

        p_nn = p_nn / p_nn.sum()

        max_alpha, pi_bar = find_max_alpha(s, self._env, self.MCTS, p_nn)

        pi_bar[mask] = 0

        pi_bar = pi_bar / pi_bar.sum()

        if self._act_max:
            max_child = np.argmax(pi_bar)
            pi_bar = np.zeros((self._env.action_size,))
            pi_bar[max_child] = 1

        random_child = np.random.choice(pi_bar,p=pi_bar)

        return np.where(pi_bar == random_child)[0][0], pi_bar

    def create_model(self,width,height,nn_type):

        if nn_type == "cnn":
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

    def display_state_info(self,state,action):
        """
        Display value of current state
        """

        player = self._env.check_turn(state)
        winner = self._env.check_winner(state)

        node = ZeroNode(state=state, player=player, winner=winner, parent=None, parent_action=action)

        self.MCTS.display_state_info(node)

    def eval(self):
        """
        Used when evaluating the agent to ensure
        the action acts greedily
        """
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

    def get_child_values(self,state):

        og_state = self.reverse_transform(state)

        valid_actions = self._env.valid_actions(og_state)

        child_states = [self._env.next_state(og_state,a) for a in valid_actions]

        child_states = [self.transform_state(c) for c in child_states]

        child_states = [torch.FloatTensor(c) for c in child_states]

        vs = [self._nn.predict(c)[1] for c in child_states]

        p = np.zeros((9,))

        p[valid_actions] = vs

        return p

    def get_policy_action(self,state):
        """
        Select best action just from policy network
        """
        state = self.process_state(state)
        state = Variable(torch.from_numpy(state).type(dtype=torch.float))
        p , v = self._nn.predict(state)

        return np.argmax(p)

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

        if root not in self.MCTS.dec_pts:

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
        state = self.process_state(state)
        state = Variable(torch.from_numpy(state).type(dtype=torch.float))

        return state

    def predict(self,s):
        state = self.node_to_state(s)
        return self._nn.predict(state)

    def process_state(self,state):

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
        state_original[p1_actions] = 1
        p2_actions = np.where(state[1].flatten() == 1)[0]

        state_original[p2_actions] = 2

        return state_original

    def run_search(self,state):

        self.current_state = state.copy()

        s = self.get_node(self.current_state)

        self.MCTS.root = s
        # First run simulations to collect
        # Tree statistics
        for _ in range(self._n_sim):
            self.MCTS.run(s)

        return s

    def save(self,id):
        torch.save(self._nn.state_dict(), f"current_best_{self._env.name}_{id}")

    def show_data_sample(self,sample):
        """
        Method to help get a visual on batch of training data
        :param sample:
        :return:
        """

        self._nn.eval()

        for data in sample:
            p, v = self._nn.predict(torch.FloatTensor(data.state))
            state = self.reverse_transform(data.state)
            print("-----------------state---------------------")
            print(state.reshape((3,3)))
            print("MCTS PROBS")
            print(data.action_mcts)
            print("NN P")
            print(p.reshape(3,3))
            print(f"True Value = {data.z}")
            print("NN V")
            print(v)
            print("Children Values")
            child_values = self.get_child_values(data.state)
            print(child_values.reshape((3,3)))

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

            self._memory.push(mem)

        self.reset_current_memory()

    def view_current_memory(self):

        last_n = 40
        data = self._memory.memory[-last_n:]

        self.show_data_sample(data)

    def train_network(self):

        if len(self._memory) < self._batch_size:
            return None,None,None

        #self.view_current_memory()
        if self._optimizer == "Adam":
            optimizer = torch.optim.Adam(self._nn.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        elif self._optimizer == "SGD":
            optimizer = torch.optim.SGD(self._nn.parameters(), lr=self._lr, momentum=self._momentum)
        else:
            print("Passed incorrect optimizer. Using SGD by default")
            optimizer = torch.optim.SGD(self._nn.parameters(), lr=self._lr, momentum=self._momentum)

        avg_total = 0
        avg_value = 0
        avg_policy = 0

        for _ in range(self._epochs):

            self._nn.train()

            data = self._memory.sample(self._batch_size)

            b, p, v = zip(*[d for d in data])

            #b, p, v = d

            target_p = torch.FloatTensor(p)
            target_v = torch.FloatTensor(v)
            #target_v = torch.FloatTensor([float(v)])
            state = torch.FloatTensor(b)

            #target_p = target_p.reshape((1,9))
            #state = state.reshape((1,3,3,3))

            p, v = self._nn(state)

            loss_value = self.loss_v(target_v,v)

            loss_policy = self.loss_pi(target_p, p)

            loss = loss_value + loss_policy #loss_policy
            """
            for i in range(len(data)):

                latent_b,mcts_p,z = data[i]
                b = self.reverse_transform(latent_b)
                print("Board")
                print(b.reshape((3,3)))
                print("Latent B")
                print(latent_b)
                print("MCTS PP")
                print(mcts_p.reshape((3,3)))
                print("Predicted PP")
                print(p[i].detach().numpy().reshape((3,3)))
                print(f"Game result = {z}")
                print(f"NN prediction = {float(v[i])}")
            """

            avg_total += loss
            avg_value += loss_value
            avg_policy += loss_policy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

    def loss_minimax(self):
        """
            Compute loss of agent actions vs minimax agents actions
        """
        s_t = time.time()
        self.eval()
        num_state = len(self._minimax_actions)
        policy_correct = 0.0
        mcts_correct = 0.0
        mcts_bar_correct = 0.0
        for k,v in self._minimax_actions.items():
            state = self._state_lookup[k]
            policy_a = self.get_policy_action(state)
            #print(state.reshape(3,3))
            minimax_action = self._minimax_actions[k]
            #print(minimax_action)
            if policy_a in minimax_action:
                policy_correct += 1
            self.reset()
            s = self.run_search(state)
            #bar_a, pi_bar = self.compute_pi_bar(s)

            mcts_a, mcts_pi = self.get_action(s)

            #if bar_a in self._minimax_actions[k]:
            #    mcts_bar_correct += 1

            if mcts_a in self._minimax_actions[k]:
                mcts_correct += 1

        e_t = time.time()

        run_time = (e_t - s_t) / 60.0

        print("Loss minimax runtime")
        print(run_time)
        self.train()
        return policy_correct / num_state , mcts_correct / num_state , mcts_bar_correct / num_state

    def loss_pi(self, targets, outputs):
        assert targets.shape == outputs.shape
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        assert targets.shape == outputs.view(-1).shape
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    @staticmethod
    def load_lookup(path) -> dict:
        if not path:
            return {}

        if len(path) > 0:
            return np.load(path, allow_pickle=True).item()
        else:
            return {}
